#ifndef IPU_BASE_HPP
#define IPU_BASE_HPP

#ifndef KLIGN_IPU_MAXAB_SIZE
#define KLIGN_IPU_MAXAB_SIZE 200
#endif

#ifndef KLIGN_IPU_TILES
#define KLIGN_IPU_TILES 8832
#endif

#ifndef KLIGN_IPU_BATCHES
#define KLIGN_IPU_BATCHES 110
#endif

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>

#include "similarity.h"
#include "encoding.h"

using namespace poplar;
namespace ipu {

struct SWConfig {
        int gapInit = 1;
        int gapExtend = 1;
        int matchValue = 1;
        int mismatchValue = -1;
        swatlib::Similarity similarity = swatlib::Similarity::nucleicAcid;
        swatlib::DataType datatype = swatlib::DataType::nucleicAcid;
};

Type formatToType(const std::string& format);

TypeTraits typeToTrait(const Type& t);

void convertSimilarityMatrix(Target& target, const Type& t, swatlib::Matrix<int8_t> matrix, void** buffer);

inline void addCodelets(Graph& graph);

inline int extractScoreSW(Engine& engine, const std::string& sA, const std::string& sB);

class IPUAlgorithm {
    poplar::Device device;
    poplar::Target target;
protected:
    SWConfig config;

    std::unique_ptr<Engine> engine;
public:
    IPUAlgorithm(SWConfig config);

    void addCycleCount(Graph& graph, program::Sequence& mainProgram);

    Graph createGraph();

    // Needs to be called in child class
    void createEngine(Graph& graph, std::vector<program::Program> programs);

    poplar::Target& getTarget();
    poplar::Device& getDevice();
    poplar::Graph getGraph();
};

}

#endif // IPU_BASE_HPP