// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
// This is graphcores Prefetch example: https://github.com/graphcore/tutorials/blob/master/feature_examples/poplar/prefetch/prefetch.cpp

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
// #include <capnp/compat/json.h>
// #include <capnp/message.h>
// #include <capnp/serialize.h>
// #include <kj/std/iostream.h>

#include <algorithm>
#include <iostream>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>


#include <cassert>
#include <cstdlib>
using namespace std;
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

  cout << "one" << endl;

  const unsigned elements = 8;
  const unsigned repeat = 5;
  const unsigned numIpus = 1;

  // Obtain list of available devices
  auto manager = DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, numIpus);
  if (devices.empty()) {
      throw poplar::poplar_error("No devices found");
  }

  cout << "two" << endl;
  // Connect to first available device
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &d){ return d.attach(); });
  if (it == devices.end()) {
      throw poplar::poplar_error("Could not attach to any device");
  }

  auto device = std::move(*it);

  cout << "three" << endl;

  // https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/tensorflow/compiling.html

  std::fstream infile;
  infile.open("./resources/exec.poplar_exec");
  cout << "four" << endl;
  if (!infile.is_open()) {
    cout << "nooooooo!" << endl;
  }

  OptionFlags options{{"exchange.streamBufferOverlap", "none"},
                    {"exchange.enablePrefetch", "true"}};

  // https://github.com/graphcore/popart/blob/15ce5b098638dc34a4d41ae2a7621003458df798/willow/src/popx/executablexserialization.cpp

  cout << "five" << endl;
  auto exe = Executable::deserialize(infile);
  cout << "six" << endl;

  Engine eng(std::move(exe), options);
  cout << "seven" << endl;
  
  eng.load(device);
  // eng.run(0);

  // Connect output
  std::vector<float> hOut(elements);
  eng.connectStream("out", hOut.data(),
                    std::next(hOut.data(), hOut.size()));

  boost::optional<float> previous;
  for (float in : {1.0f, 3.0f, 3.0f, 9.0f, 9.0f}) {
    if (previous != in) {
      std::cout << "Set input callback\n";
      std::unique_ptr<FillCallback> cb{new FillCallback(in, elements)};
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


  return 0;
}

#endif
