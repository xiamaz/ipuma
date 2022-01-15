add_gp_library(
  NAME algoipu
  ASM_SOURCES
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/sw.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/codelet_batch_affine.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/helper.hpp
)

add_library(IPUMA_LIBRARY "ipu_batch_affine.cpp" "matrix.cpp" "ipu_base.cpp" "similarity.cpp" "encoding.cpp")
target_link_libraries(IPUMA_LIBRARY poplar popops poputil)
add_dependencies(IPUMA_LIBRARY  algoipu.gp_codelets)