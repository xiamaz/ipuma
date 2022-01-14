add_gp_library(
  NAME algoipu
  ASM_SOURCES
  CPP_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/sw.cpp
  HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/codelets/helper.hpp
)