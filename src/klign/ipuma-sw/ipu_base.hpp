#ifndef IPU_BASE_HPP
#define IPU_BASE_HPP

#include <string>

#include <libgen.h>
#include <unistd.h>

#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/SyncType.hpp>
#include <poplar/CycleCount.hpp>
// #include <poplar/IPUModel.hpp>
#include <poplar/Program.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/VariableMappingMethod.hpp>

#include <poputil/TileMapping.hpp>

#include "popops/ElementWise.hpp"
#include "popops/codelets.hpp"

#include "ipulib.hpp"
#include "matrix.h"
#include "similarity.h"
#include "encoding.h"

using namespace poplar;
namespace ipu {

struct SWConfig {
        int gapInit = 1;
        int gapExtend = 1;
        int matchValue = 2;
        int mismatchValue = -2;
        swatlib::Similarity similarity = swatlib::Similarity::simple;
        swatlib::DataType datatype = swatlib::DataType::string;
};

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

class IPUAlgorithm : IPUContext {
protected:
    SWConfig config;

    std::unique_ptr<Engine> engine;
    int activeTiles, bufSize;
public:
    IPUAlgorithm(SWConfig config, int bufSize = 10001, int activeTiles = 1472) : config(config), activeTiles(activeTiles), bufSize(bufSize) {
    }

    /**
     * Check that the array fits into the ipu.
     */
    bool checkSize(const std::vector<std::string>& S) {
        int maxLen = 0;
        for (const std::string& s : S) {
            int sSize = s.size();
            if (sSize > maxLen) maxLen = sSize;
        }
        return maxLen < bufSize;
    }

    void addCycleCount(Graph& graph, program::Sequence& mainProgram) {
        Tensor cycles = cycleCount(graph, mainProgram, 1, SyncType::EXTERNAL);
        graph.createHostRead("cycles", cycles);
    }

    Graph createGraph() {
        auto target = getTarget();
        Graph graph(target);
        addCodelets(graph);
        popops::addCodelets(graph);
        return graph;
    }

    // Needs to be called in child class
    void createEngine(Graph& graph, std::vector<program::Program> programs) {
        auto& device = getDevice();
        poplar::OptionFlags engineOptions;
        std::cout << "Help1\n";
        engine = std::make_unique<Engine>(graph, programs, engineOptions);
        std::cout << "Help2\n";
        engine->load(device);
    }

    template <typename Dest, typename Source>
    void writeData(const std::string& dest, typename std::vector<std::vector<Source>>::iterator begin, typename std::vector<std::vector<Source>>::iterator end, size_t destSize, size_t rowSize, Dest fillValue = 0) {
        size_t bufferSize = destSize * sizeof(Dest);
        Dest* buffer = (Dest*) malloc(bufferSize);
        memset(buffer, fillValue, bufferSize);
        size_t rowBufferSize = rowSize * sizeof(Dest);
        int i = 0;
        for (auto rt = begin; rt != end; ++rt) {
            for (int j = 0; j < (*rt).size(); ++j) {
                buffer[i * rowBufferSize + j] = (*rt)[j];
            }
            ++i;
        }
        engine->writeTensor(dest, buffer, buffer + bufferSize);
        free(buffer);
    }

    template <typename Dest, typename Source>
    void writeData(const std::string& dest, const std::vector<std::vector<Source>>& data, size_t destSize, size_t rowSize, Dest fillValue = 0) {
        size_t bufferSize = destSize * sizeof(Dest);
        Dest* buffer = (Dest*) malloc(bufferSize);
        memset(buffer, fillValue, bufferSize);
        for (int i = 0; i < data.size(); ++i) {
            const std::vector<Source>& row = data[i];
            for (int j = 0; j < row.size(); ++j) {
                size_t rowBufferSize = rowSize * sizeof(Dest);
                buffer[i * rowBufferSize + j] = row[j];
            }
        }
        engine->writeTensor(dest, buffer, buffer + bufferSize);
        free(buffer);
    }

    template <typename Dest, typename Source>
    void writeData(const std::string& dest, const std::vector<Source>& data, size_t destSize) {
        std::vector<std::vector<Source>> vdata = {data};
        writeData<Dest, Source>(dest, vdata, destSize, destSize);
    }

    template <typename Dest>
    void writeData(const std::string& dest, const std::vector<std::string>& data, size_t destSize, size_t rowSize) {
        std::vector<std::vector<char>> cdata;
        for (auto& d : data) {
            cdata.push_back(std::vector<char>(d.begin(), d.end()));
        }
        writeData<Dest, char>(dest, cdata, destSize, rowSize);
    }

    template <typename Dest>
    void writeData(const std::string& dest, const std::string& data, size_t destSize) {
        std::vector<std::string> vdata = {data};
        writeData<Dest>(dest, vdata, destSize, destSize);
    }

    /**
     * Write one string into a duplicated buffer.
     */
    template<typename Dest, typename Source>
    void writeDuplicate(const std::string& dest, const std::vector<Source>& data, size_t destSize, size_t rowSize, size_t count, Dest fillValue = 0) {
        size_t bufferSize = destSize * sizeof(Dest);
        Dest* buffer = (Dest*) malloc(bufferSize);
        memset(buffer, fillValue, bufferSize);
        for (int i = 0; i < count; ++i) {
            for (int j =0; j < data.size(); ++j) {
                size_t rowBufferSize = rowSize * sizeof(Dest);
                buffer[i * rowBufferSize + j] = data[j];
            }
        }
        engine->writeTensor(dest, buffer, buffer + bufferSize);
        free(buffer);
    }

    template<typename T>
    void writeDuplicate(const std::string& dest, const std::string& data, size_t destSize, size_t rowSize, size_t count) {
        std::vector<char> cdata(data.begin(), data.end());
        writeDuplicate<T, char>(dest, cdata, destSize, rowSize, count);
    }
};

}

#endif // IPU_BASE_HPP