#include <linux/limits.h>
#include <string>
#include <stdexcept>

#include <libgen.h>
#include <unistd.h>

#include <poplar/Engine.hpp>
#include <poplar/SyncType.hpp>
#include <poplar/CycleCount.hpp>
// #include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/VariableMappingMethod.hpp>

#include <poputil/TileMapping.hpp>

#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"

#include "matrix.h"
#include "similarity.h"
#include "encoding.h"

#include "ipu_base.h"

using namespace poplar;
namespace ipu {

Type formatToType(const std::string& format) {
    if (format == "int") {
        return INT;
    } else if (format == "float") {
        return FLOAT;
    } else if (format == "half") {
        return HALF;
    } else if (format == "short") {
        return SHORT;
    } else {
        throw std::runtime_error("Unknown type format: " + format);
    }
}

TypeTraits typeToTrait(const Type& t) {
    if (t == INT) {
        return TypeTraits::make<int32_t>();
    } else if (t == FLOAT) {
        return TypeTraits::make<float>();
    } else if (t == HALF) {
        return TypeTraits::make<IeeeHalf>();
    } else if (t == SHORT) {
        return TypeTraits::make<int16_t>();
    } else {
        throw std::runtime_error("Unsupported type for trait conversion: " + t.toString());
    }
}

void convertSimilarityMatrix(Target& target, const Type& t, swatlib::Matrix<int8_t> matrix, void** buffer) {
    auto [m, n] = matrix.shape();
    void* b;
    int8_t* matrixp = matrix.data();
    if (t == INT) {
        b = malloc(m * n * sizeof(int32_t));
        auto s = static_cast<int32_t*>(b);
        for (int i = 0; i < m * n; ++i) {
            s[i] = static_cast<int32_t>(matrixp[i]);
        }
    } else if (t == FLOAT) {
        b = malloc(m * n * sizeof(float));
        auto s = static_cast<float*>(b);
        for (int i = 0; i < m * n; ++i) {
            s[i] = static_cast<float>(matrixp[i]);
        }
    } else if (t == HALF) {
        b = malloc(m * n * 2);
        float* fbuf = (float*) malloc(m * n * sizeof(float));
        for (int i = 0; i < m * n; ++i) {
            fbuf[i] = static_cast<float>(matrixp[i]);
        }
        copyFloatToDeviceHalf(target, fbuf, b, m * n);
        free(fbuf);
    } else if (t == SHORT) {
        b = malloc(m * n * sizeof(int16_t));
        auto s = static_cast<int16_t*>(b);
        for (int i = 0; i < m * n; ++i) {
            s[i] = static_cast<int16_t>(matrixp[i]);
        }
    } else {
        throw std::runtime_error("Unknown type format: " + t.toString());
    }
    *buffer = b;
}

std::string get_selfpath() {
    char buff[PATH_MAX];
    ssize_t len = ::readlink("/proc/self/exe", buff, sizeof(buff)-1);
    if (len != -1) {
      buff[len] = '\0';
      return std::string(buff);
    }
    return "";
}

inline void addCodelets(Graph& graph) {
    auto selfPath = get_selfpath();
    auto rootPath = std::string(dirname(dirname(&selfPath[0])));
    std::cout << rootPath << "\n";
    graph.addCodelets(rootPath + "/bin/codelets/algoipu.gp");
}

inline int extractScoreSW(Engine& engine, const std::string& sA, const std::string& sB) {
    unsigned long m = sA.length() + 1;
    unsigned long n = sB.length() + 1;

    swatlib::Matrix<short> S(m, n);
    swatlib::Matrix<char> D(m, n);

    engine.readTensor("s-read", &*S.begin(), &*S.end());
    engine.readTensor("d-read", &*D.begin(), &*D.end());

    // std::cout << S.toString() << std::endl;
    auto [x, y] = S.argmax();
    return S(x, y);
}

IPUAlgorithm::IPUAlgorithm(SWConfig config) : config(config) {
    auto manager = poplar::DeviceManager::createDeviceManager();
    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);
    std::cout << "Trying to attach to IPU\n";
    auto it = std::find_if(devices.begin(), devices.end(), [](poplar::Device &device) {
       return device.attach();
    });

    if (it == devices.end()) {
      std::cerr << "Error attaching to device\n";
    }

    device = std::move(*it);
    target = device.getTarget();
}

void IPUAlgorithm::addCycleCount(Graph& graph, program::Sequence& mainProgram) {
    Tensor cycles = cycleCount(graph, mainProgram, 1, SyncType::EXTERNAL);
    graph.createHostRead("cycles", cycles);
}

Graph IPUAlgorithm::createGraph() {
    auto target = getTarget();
    Graph graph(target);
    addCodelets(graph);
    popops::addCodelets(graph);
    return graph;
}

// Needs to be called in child class
void IPUAlgorithm::createEngine(Graph& graph, std::vector<program::Program> programs) {
    auto& device = getDevice();
    poplar::OptionFlags engineOptions;
    engine = std::make_unique<Engine>(graph, programs, engineOptions);
    engine->load(device);
}

poplar::Target& IPUAlgorithm::getTarget() {
          return target;
}

poplar::Device& IPUAlgorithm::getDevice() {
                return device;
}

poplar::Graph IPUAlgorithm::getGraph() {
          return std::move(poplar::Graph(target));
}

}