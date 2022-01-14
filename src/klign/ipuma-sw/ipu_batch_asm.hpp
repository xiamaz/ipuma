#ifndef IPU_BATCH_ASM_HPP
#define IPU_BATCH_ASM_HPP

// Smith Waterman with static graph size.
#include <string>
#include <cmath>

#include <nlohmann/json.hpp>

#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/TypeTraits.hpp>

#include <popops/Zero.hpp>

#include "ipulib/ipulib.hpp"
#include "ipu_base.hpp"

using namespace poplar;
using json = nlohmann::json;

namespace algo {
namespace ipu {
namespace batchasm {
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
    Tensor Scores = graph.addVariable(INT, {activeTiles}, "Scores");

    poplar::Type sType = formatToType(format);
    TypeTraits traits = typeToTrait(sType);
    void* similarityBuffer;
    convertSimilarityMatrix(target, sType, similarityData, &similarityBuffer);
    Tensor similarity = graph.addConstant(sType, {m * n}, similarityBuffer, traits, false, "similarity");
    free(similarityBuffer);

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
    graph.createHostWrite("alen-write", Alens);
    graph.createHostWrite("b-write", Bs);
    graph.createHostWrite("blen-write", Blens);
    graph.createHostRead("scores-read", Scores);

    auto frontCs = graph.addComputeSet("front");
    for (int i = 0; i < activeTiles; ++i) {
        int tileIndex = i % tileCount;
        VertexRef vtx = graph.addVertex(
            frontCs, "SWAsm<" + format + ">",
            {
                {"A", As[i]},
                {"Alen", Alens[i]},
                {"B", Bs[i]},
                {"Blen", Blens[i]},
                {"simMatrix", similarity},
                {"simWidth", m},
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
    SWAlgorithm(const json& args) : IPUAlgorithm(args) {
        format = args.value("format", "float");
        auto similarityMatrix = swatlib::selectMatrix(similarity, matchValue, mismatchValue);

        Graph graph = createGraph();

        std::vector<program::Program> programs = buildGraph(graph, activeTiles, bufSize, format, similarityMatrix);

        if (devel) {
            addCycleCount(graph, static_cast<program::Sequence&>(programs[0]));
        }
        
        createEngine(graph, programs);
    }

    bool hasVectorizedCompare() { return true; }

    swatlib::Matrix<int> compare(const std::vector<std::string>& A, const std::vector<std::string>& B, swatlib::TickTock& t) {
        if (!(checkSize(A) || checkSize(B))) throw std::runtime_error("Too small buffer or number of active tiles.");
        size_t transSize = activeTiles * bufSize * sizeof(uint32_t);

        auto encoder = getEncoder(datatype);
        auto vA = encoder.encode(A);
        auto vB = encoder.encode(B);

        uint64_t totalCycles = 0;
        swatlib::Matrix<int> results(vA.size(), vB.size());

        size_t totalBufferSize = activeTiles * bufSize * sizeof(uint8_t);
        size_t rowBufferSize = bufSize * sizeof(uint8_t);
        uint8_t* bufferA = (uint8_t*) malloc(totalBufferSize);
        uint8_t* bufferB = (uint8_t*) malloc(totalBufferSize);
        size_t lenBufferSize = activeTiles * sizeof(uint32_t);
        uint32_t* bufferAlen = (uint32_t*) malloc(lenBufferSize);
        uint32_t* bufferBlen = (uint32_t*) malloc(lenBufferSize);

        int bi = 0, ai =0;
        // always fit the maximum number of comparisons onto all tiles
        t.tick();
        while (bi < vB.size() && ai < vA.size()) {
            std::vector<std::tuple<int, int>> indices;
            std::vector<int> scores(activeTiles);

            memset(bufferA, 0, totalBufferSize);
            memset(bufferB, 0, totalBufferSize);
            memset(bufferAlen, 0, lenBufferSize);
            memset(bufferBlen, 0, lenBufferSize);

            for (int k = 0; k < activeTiles; ++k) {
                auto& a = vA[ai];
                auto& b = vB[bi];
                for (int l = 0; l < a.size(); ++l) {
                    bufferA[k * rowBufferSize + l] = a[l];
                }
                for (int l = 0; l < b.size(); ++l) {
                    bufferB[k * rowBufferSize + l] = b[l];
                }
                bufferAlen[k] = a.size();
                bufferBlen[k] = b.size();

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
            engine->writeTensor("alen-write", bufferAlen, bufferAlen + lenBufferSize);
            engine->writeTensor("blen-write", bufferBlen, bufferBlen + lenBufferSize);

            engine->run(0);

            if (devel) {
                uint32_t cycles[2];
                engine->readTensor("cycles", &cycles, &cycles + 8);
                totalCycles += (((uint64_t)cycles[1]) << 32) | cycles[0];
            }
            engine->readTensor("scores-read", &*scores.begin(), &*scores.end());
            for (int k = 0; k < indices.size(); ++k) {
                auto [i, j] = indices[k];
                results(i, j) = scores[k];
            }
        }
        t.tock();

        log["cycles"] = totalCycles;

        free(bufferA);
        free(bufferB);
        free(bufferAlen);
        free(bufferBlen);

        return results;
    }
};
}
}
}
#endif // IPU_BATCH_ASM_HPP