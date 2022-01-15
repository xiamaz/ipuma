#ifndef IPU_BATCH_AFFINE_HPP
#define IPU_BATCH_AFFINE_HPP

// Smith Waterman with static graph size.
#include <string>
#include <cmath>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>

#include <popops/Zero.hpp>

#include "ipulib.hpp"
#include "ipu_base.hpp"
#include "similarity.h"
#include "encoding.h"

using namespace poplar;

namespace ipu {
namespace batchaffine {
/**
 * Streamable IPU graph for SW
 */
std::vector<program::Program> buildGraph(Graph& graph, unsigned long activeTiles, unsigned long bufSize, const std::string& format, const swatlib::Matrix<int8_t> similarityData) {
    program::Sequence prog;
    program::Sequence initProg;

    auto target = graph.getTarget();
    int tileCount = target.getTilesPerIPU();

    Tensor As = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "A");
    Tensor Bs = graph.addVariable(UNSIGNED_CHAR, {activeTiles, bufSize}, "B");

    Tensor Alens = graph.addVariable(UNSIGNED_INT, {activeTiles}, "Alen");
    Tensor Blens = graph.addVariable(UNSIGNED_INT, {activeTiles}, "Blen");

    auto [m, n] = similarityData.shape();

    poplar::Type sType = formatToType(format);
    TypeTraits traits = typeToTrait(sType);
    void* similarityBuffer;
    convertSimilarityMatrix(target, sType, similarityData, &similarityBuffer);
    Tensor similarity = graph.addConstant(sType, {m, n}, similarityBuffer, traits, false, "similarity");
    free(similarityBuffer);

    // Tensor similarity = graph.addConstant(SIGNED_CHAR, {m, n}, similarityData.data(), "similarity");
    Tensor Scores = graph.addVariable(INT, {activeTiles}, "Scores");

    graph.setTileMapping(similarity, 0);
    for (int i = 0; i < activeTiles; ++i) {
        int tileIndex = i % tileCount;
        graph.setTileMapping(As[i], tileIndex);
        graph.setTileMapping(Bs[i], tileIndex);
        graph.setTileMapping(Scores[i], tileIndex);
        graph.setTileMapping(Alens[i], tileIndex);
        graph.setTileMapping(Blens[i], tileIndex);
    }

    graph.createHostWrite("a-write", As);
    graph.createHostWrite("b-write", Bs);
    graph.createHostWrite("alen-write", Alens);
    graph.createHostWrite("blen-write", Blens);
    graph.createHostRead("scores-read", Scores);

    auto frontCs = graph.addComputeSet("front");
    for (int i = 0; i < activeTiles; ++i) {
        int tileIndex = i % tileCount;
        VertexRef vtx = graph.addVertex(
            frontCs, "SWAffine<" + format + ">",
            {
                {"A", As[i]},
                {"B", Bs[i]},
                {"simMatrix", similarity},
                {"bufSize", bufSize},
                {"score", Scores[i]},
                {"gapInit", 0},
                {"gapExt", -1},
            }
        );
        graph.setFieldSize(vtx["C"], bufSize + 1);
        graph.setFieldSize(vtx["bG"], bufSize + 1);
        graph.setTileMapping(vtx, tileIndex);
        graph.setPerfEstimate(vtx, 1);
    }
    prog.add(program::Execute(frontCs));
    return {prog, initProg};
}

class SWAlgorithm : public IPUAlgorithm {
protected:
    std::string format;
public:
    SWAlgorithm(SWConfig config, IPUContext &ctx, int bufSize = 10001, int activeTiles = 1472) : IPUAlgorithm(config, ctx, activeTiles, bufSize) {
        auto similarityMatrix = swatlib::selectMatrix(config.similarity, config.matchValue, config.mismatchValue);

        Graph graph = createGraph();

        std::vector<program::Program> programs = buildGraph(graph, activeTiles, bufSize, format, similarityMatrix);

        createEngine(graph, programs);
    }

    bool hasVectorizedCompare() { return true; }

    swatlib::Matrix<int> compare(const std::vector<std::string>& A, const std::vector<std::string>& B) {
        if (!(checkSize(A) || checkSize(B))) throw std::runtime_error("Too small buffer or number of active tiles.");
        size_t transSize = activeTiles * bufSize * sizeof(char);

        auto encoder = swatlib::getEncoder(config.datatype);
        auto vA = encoder.encode(A);
        auto vB = encoder.encode(B);

        uint64_t totalCycles = 0;
        swatlib::Matrix<int> results(vA.size(), vB.size());

        size_t totalBufferSize = activeTiles * bufSize * sizeof(uint8_t);
        size_t rowBufferSize = bufSize * sizeof(uint8_t);
        uint8_t* bufferA = (uint8_t*) malloc(totalBufferSize);
        uint8_t* bufferB = (uint8_t*) malloc(totalBufferSize);

        int bi = 0, ai =0;
        // always fit the maximum number of comparisons onto all tiles
        while (bi < vB.size() && ai < vA.size()) {
            std::vector<std::tuple<int, int>> indices;
            std::vector<int> scores(activeTiles);

            memset(bufferA, encoder.get_terminator(), totalBufferSize);
            memset(bufferB, encoder.get_terminator(), totalBufferSize);

            for (int k = 0; k < activeTiles; ++k) {
                auto& a = vA[ai];
                auto& b = vB[bi];
                for (int l = 0; l < a.size(); ++l) {
                    bufferA[k * rowBufferSize + l] = a[l];
                }
                for (int l = 0; l < b.size(); ++l) {
                    bufferB[k * rowBufferSize + l] = b[l];
                }

                indices.push_back({ai, bi});
                // increment indices
                ++bi;
                if (bi >= vB.size()) {
                    ++ai;
                    bi = 0;
                }
                if (ai >= vA.size()) {
                    break;
                }
            }

            engine->writeTensor("a-write", bufferA, bufferA + totalBufferSize);
            engine->writeTensor("b-write", bufferB, bufferB + totalBufferSize);

            engine->run(0);

            engine->readTensor("scores-read", &*scores.begin(), &*scores.end());
            for (int k = 0; k < indices.size(); ++k) {
                auto [i, j] = indices[k];
                results(i, j) = scores[k];
            }
        }

        free(bufferA);
        free(bufferB);

        return results;
    }
};
}
}
#endif // IPU_BATCH_AFFINE_HPP