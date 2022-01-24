add_gp_library(
  NAME algoipu
  ASM_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/codelet_batch_asm.S
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/codelet_batch_affine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/codelet_batch_asm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/codelet_multi_affine.cpp
  HEADERS
)

add_library(IPUMA_LIBRARY "ipu_batch_affine.cpp" "matrix.cpp" "ipu_base.cpp" "similarity.cpp" "encoding.cpp" "driver.cpp")
target_link_libraries(IPUMA_LIBRARY poplar popops poputil ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
add_dependencies(IPUMA_LIBRARY  algoipu.gp_codelets)