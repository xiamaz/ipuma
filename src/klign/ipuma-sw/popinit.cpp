// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

/* This file contains the completed version of Poplar tutorial 3.
   See the Poplar user guide for details.
*/

#include <iostream>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poplar/IPUModel.hpp>
#include <string>
// #include <filesystem>

#include "popinit.hpp"


using namespace poplar;
using namespace poplar::program;

// inline std::filesystem::path getCodeletPath() {
//     auto exDir = getExecutablePath().parent_path() / "codelets";
//     return exDir;
// }


inline void addCodelets(Graph& graph, const std::string& codeletName) {
//     const std::string codeletPath = (getCodeletPath() / codeletName).string();
    graph.addCodelets("/home/lukb/git/isc21/ipuma/build/bin/codelets/algoipu.gp");
}

int mainx() {
  // Create the IPU model device
  IPUModel ipuModel;
  Device device = ipuModel.createDevice();
  Target target = device.getTarget();

  // Create the Graph object
  Graph graph(target);
  graph.addCodelets("/home/lukb/git/isc21/ipuma/build/bin/codelets/algoipu.gp");

  // Add codelets to the graph


  // Add variables to the graph
  Tensor v1 = graph.addVariable(FLOAT, {4}, "v1");
  Tensor v2 = graph.addVariable(FLOAT, {4}, "v2");
  for (unsigned i = 0; i < 4; ++i) {
    graph.setTileMapping(v1[i], i);
    graph.setTileMapping(v2[i], i);
  }

  // Create a control program that is a sequence of steps
  Sequence prog;

  // Add steps to initialize the variables
  Tensor c1 = graph.addConstant<float>(FLOAT, {4}, {1.0, 1.5, 2.0, 2.5});
  graph.setTileMapping(c1, 0);
  prog.add(Copy(c1, v1));

  ComputeSet computeSet = graph.addComputeSet("computeSet");
  for (unsigned i = 0; i < 4; ++i) {
    VertexRef vtx = graph.addVertex(computeSet, "Testoo");
    graph.setTileMapping(vtx, i);
    graph.setPerfEstimate(vtx, 20);
  }

  // Add step to execute the compute set
  prog.add(Execute(computeSet));

  // Create the engine
  Engine engine(graph, prog);
  engine.load(device);

  // Run the control program
  std::cout << "Running program\n";
  engine.run(0);
  std::cout << "Program complete\n";

  return 0;
}