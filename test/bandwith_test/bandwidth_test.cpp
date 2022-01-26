#ifdef __POPC__

#include <poplar/Vertex.hpp>

using namespace poplar;

class EmptyVertex : public Vertex {
public:
  Vector<Input<Vector<unsigned char>>> x;
  // Input<Vector<unsigned char>> x;
  bool compute() {
    return true;
  }
};

#else

#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/exceptions.hpp>

#include <linux/limits.h>
#include <string>
#include <stdexcept>

#include <libgen.h>
#include <unistd.h>

#include <poplar/Engine.hpp>
#include <poplar/SyncType.hpp>
#include <poplar/CycleCount.hpp>
#include <poplar/Program.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/VariableMappingMethod.hpp>

#include <poputil/TileMapping.hpp>

#include <boost/optional.hpp>

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <cassert>
#include <cstdlib>

using namespace poplar;
using namespace poplar::program;

// Benchmark description: we measure aggregate bandwidth with a benchmark
// that transfers data from the host to a tensor on IPUs via graph::DataStream.
// The destination tensor is partitioned linearly across all involved tiles and
// IPUs. The benchmark sends 40 KB data to each tile, and up to 778.24 MB
// to all 16 IPUs. We report results in Table 4.20.

// https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html#data-streams-and-remote-buffers

// For a host-to-device stream, by default the callback function will be 
// called immediately before the IPU transfers the buffer contents to 
// device memory. The host-side code should populate the stream buffer 
// and then return.

// Note that if buffers are not overlapped in host memory (see the Engine option 
// exchange.streamBufferOverlap) then stream splitting is not possible. This is 
// because a buffer is allocated exclusively to a data stream, independently of 
// sync IDs, and so nothing can be done to reduce the buffer usage.


// Callback
class Callback final : public poplar::StreamCallback {
public:
  using Result = poplar::StreamCallback::Result;

  Callback() {}

  Result prefetch(void *__restrict p) noexcept override {
    std::cout << "prefetch\n";
    return Result::Success;
  }

  void fetch(void *__restrict p) noexcept override {
    std::cout << "fetch\n";
  }

  void complete() noexcept override {
    std::cout << "complete\n";
  }

private:
  float value;
  size_t size;
};

void addCycleCount(Graph& graph, program::Sequence& mainProgram, const std::string handle) {
    Tensor cycles = cycleCount(graph, mainProgram, 1, SyncType::EXTERNAL);
    graph.createHostRead(handle, cycles);
}

uint64_t getTotalCycles(Engine& engine, const std::string handle) {
  uint32_t cycles[2];
  engine.readTensor(handle, &cycles, &cycles + 1);
  uint64_t totalCycles = (((uint64_t)cycles[1]) << 32) | cycles[0];
  return totalCycles;
}

int main(int argc, char *argv[]) {

  static const std::string CYCLE_COUNT_OUTER = "cycle-count-outer";
  const unsigned ipus = 1;
  const unsigned elements = 40000;
  
  // Connect
  auto manager = DeviceManager::createDeviceManager();
  auto devices = manager.getDevices(poplar::TargetType::IPU, ipus);
  if (devices.empty()) { throw poplar::poplar_error("No devices found"); }
  auto it = std::find_if(devices.begin(), devices.end(), [](Device &d){ return d.attach(); });
  if (it == devices.end()) { throw poplar::poplar_error("Could not attach to any device"); }
  auto device = std::move(*it);

  const unsigned tiles = device.getTarget().getNumTiles();

  // Setup
  Graph graph(device);
  Tensor t = graph.addVariable(UNSIGNED_CHAR, {tiles, elements}, "data");

  for (size_t i = 0; i < tiles; i++) {
    graph.setTileMapping(t[i], i);
  }
  
  graph.addCodelets(__FILE__);
  ComputeSet cs = graph.addComputeSet();
  auto v1 = graph.addVertex(cs, "EmptyVertex");

  for (size_t i = 0; i < tiles; i++) {
    graph.setTileMapping(v1, i);
  }
  
  graph.connect(v1["x"], t);
  auto f = graph.getTarget().getTileClockFrequency();


  std::cout << "ELEMENTS: " << t.numElements() << "\n";
  auto inStream = graph.addHostToDeviceFIFO("in", UNSIGNED_CHAR, t.numElements());
  auto prog = Sequence({Copy(inStream, t.flatten())});

  addCycleCount(graph, prog, CYCLE_COUNT_OUTER);

  // Engine
  OptionFlags options{{"exchange.streamBufferOverlap", "none"},
                      {"exchange.enablePrefetch", "true"}};
  Engine eng(graph, prog, options);
  eng.load(device);

  // for (size_t i = 0; i < 1; i++) {
  std::unique_ptr<Callback> cb{new Callback()};
  eng.connectStreamToCallback("in", std::move(cb));
  std::cout << "Running\n";
  eng.run(0);

  auto cyclesOuter = getTotalCycles(eng, CYCLE_COUNT_OUTER);
  std::cout << "COUNT: " << cyclesOuter << "\n";
  // auto cyclesInner = getTotalCycles(*engine, CYCLE_COUNT_INNER);
  auto timeOuter = static_cast<double>(cyclesOuter) / device.getTarget().getTileClockFrequency();
  std::cout << "TIME: " << timeOuter << "\n";
  auto transferBandwidth = static_cast<double>(elements) / timeOuter / 1e6;
  std::cout << "BANDWITH: " << transferBandwidth << "\n";
  std::cout << "BANDWITH TILE: " << (transferBandwidth/device.getTarget().getNumTiles()) << "\n";  

  return 0;
}

#endif