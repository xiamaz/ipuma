// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// This is graphcores Prefetch example: https://github.com/graphcore/tutorials/blob/master/feature_examples/poplar/prefetch/prefetch.cpp

// GOAL: we want to perform a single compilation in order to 
// save the time that the compilation takes in the beginning 
// of each run. This is an example/playground in which we 
// try to serialize one of the poplar tutorials.

// Serialization for Poplar Graph
// https://docs.graphcore.ai/projects/poplar-api/en/latest/poplar/utility/SerializationFormat.html
// There are three functions we should make use of 
// 
// 1. Graph::serializeTensors 
//      --> https://bit.ly/34838V7
// 2. Graph::deserializeTensors 
//      --> https://bit.ly/3IvqVNG
// 3. Graph::serialize
//      --> https://bit.ly/3rLp8gN
//
// Tensor can be serializes to ether JSON or CapnProto
//
#ifdef __POPC__

#include <poplar/Vertex.hpp>

using namespace poplar;

class Increment : public Vertex {
public:
  Input<Vector<float>> x;
  Output<Vector<float>> y;

  bool compute() {
    if (x->size() != y->size())
      return false;

    for (unsigned i = 0; i < x->size(); i++) {
      y[i] = x[i] + 1;
    }
    return true;
  }
};

#else

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>
#include <poplar/Graph.hpp>
#include <poplar/Executable.hpp>
#include <poplar/SerializationFormat.hpp>
#include <poplar/DebugContext.hpp>

#include <boost/optional.hpp>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>


#include <cassert>
#include <cstdlib>

using namespace pva;
using namespace poplar;
using namespace poplar::program;

// Callback that fills the buffer with a value
class FillCallback final : public poplar::StreamCallback {
public:
  using Result = poplar::StreamCallback::Result;

  FillCallback(float value, size_t elements) : value(value), size(elements) {}

  Result prefetch(void *__restrict p) noexcept override {
    std::printf("%s (value: %g)\n", __func__, value);
    std::fill_n(static_cast<float *>(p), size, value);
    return Result::Success;
  }

  void fetch(void *__restrict p) noexcept override {
    std::printf("%s (value: %g)\n", __func__, value);
    std::fill_n(static_cast<float *>(p), size, value);
  }

  void complete() noexcept override {
    std::printf("%s (value: %g)\n", __func__, value);
  }

private:
  float value;
  size_t size;
};

std::ostream &operator<<(std::ostream &os, const std::vector<float> &values) {
  const char *sep = "";
  os << '[';
  for (auto &v : values) {
    os << sep << v;
    sep = ", ";
  }
  os << ']';
  return os;
}

int main(int argc, char *argv[]) {
  const unsigned elements = 8;
  const unsigned repeat = 5;
  const unsigned numIpus = 1;

  // Obtain list of available devices
  auto manager = DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, numIpus);
  if (devices.empty()) {
    throw poplar::poplar_error("No devices found");
  }

  // Connect to first available device
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &d){ return d.attach(); });
  if (it == devices.end()) {
    throw poplar::poplar_error("Could not attach to any device");
  }

  auto device = std::move(*it);

  // Graph elements
  Graph graph(device);
  Tensor t = graph.addVariable(FLOAT, {elements}, "data");
  graph.setTileMapping(t, 0);

  graph.addCodelets(__FILE__);

  ComputeSet cs = graph.addComputeSet();
  auto v1 = graph.addVertex(cs, "Increment");
  graph.setTileMapping(v1, 0);

  graph.connect(v1["x"], t);
  graph.connect(v1["y"], t);

  // Streams
  auto inStream = graph.addHostToDeviceFIFO("in", FLOAT, t.numElements());
  auto outStream = graph.addDeviceToHostFIFO("out", FLOAT, t.numElements());

  // Program
  auto prog = Repeat(
      repeat, Sequence({Copy(inStream, t), Execute(cs), Copy(t, outStream)}));

  // Compile program
  OptionFlags options{{"exchange.streamBufferOverlap", "none"},
                      {"exchange.enablePrefetch", "true"}};
  Engine eng(graph, prog, options);
  eng.load(device);

  // Connect output
  std::vector<float> hOut(t.numElements());
  eng.connectStream("out", hOut.data(),
                    std::next(hOut.data(), hOut.size()));

  // Run the program multiple times.
  // Replace input stream callback when value changes.
  boost::optional<float> previous;
  for (float in : {1.0f, 3.0f, 3.0f, 9.0f, 9.0f}) {
    if (previous != in) {
      std::cout << "Set input callback\n";
      std::unique_ptr<FillCallback> cb{new FillCallback(in, t.numElements())};
      eng.connectStreamToCallback("in", std::move(cb));
    }

    std::cout << "Running\n";
    eng.run(0);

    // Values in output result should be 'in' plus 1
    float expected = in + 1;
    auto match_expected =
        std::bind(std::equal_to<float>(), std::placeholders::_1, expected);
    bool success = std::all_of(hOut.begin(), hOut.end(), match_expected);
    if (!success) {
      std::stringstream ss("Unexpected result. ");
      ss << "Expected: " << expected << ";\n"
         << "Actual : " << hOut << "\n";
      throw std::runtime_error(ss.str());
    }

    previous = in;
  }

  // ------------------------------------------------------------------------------------
  // SERIALIZE TENSOR
  // ------------------------------------------------------------------------------------
  std::ofstream tensorFileJson;
  tensorFileJson.open ("./resources/tensor.json");
  graph.serializeTensors(tensorFileJson, { t }, SerializationFormat::JSON);

  std::ofstream tensorFileCbor;
  tensorFileCbor.open ("./resources/tensor.cbor");
  graph.serializeTensors(tensorFileCbor, { t }, SerializationFormat::Binary);

  // ------------------------------------------------------------------------------------
  // SERIALIZE GRAPH 
  // ------------------------------------------------------------------------------------
  std::ofstream graphFileJson;
  graphFileJson.open ("./resources/graph.json");
  graph.serialize(graphFileJson, SerializationFormat::JSON);

  std::ofstream graphFileCbor;
  graphFileCbor.open ("./resources/graph.cbor");
  graph.serialize(graphFileCbor, SerializationFormat::Binary);

  // ------------------------------------------------------------------------------------
  // EXECUTABLE
  // ------------------------------------------------------------------------------------
  OptionFlags optionss{{"exchange.streamBufferOverlap", "none"},
                      {"exchange.enablePrefetch", "true"}};

  std::function<void(int, int)> progresFunc;
  Executable executable = compileGraph(graph, 
  { prog },
  // https://bit.ly/3qXCCab
  optionss
  // progresFunc,
  // DebugContext{}
  );

  std::ofstream execFileCbor;
  execFileCbor.open ("./resources/exec.poplar_exec");
  executable.serialize(execFileCbor);
  execFileCbor.close();

  return 0;
}

#endif