set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(UPCXX REQUIRED)
find_program(UPCXX_EXEC upcxx)
set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})

set(CGRAPH_TARGET_OBJECTS)

foreach(tgt cgraph build_ctg_graph ctg_graph walk_ctg_graph)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  list(APPEND CGRAPH_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
endforeach()

add_library(CGRAPH_LIBRARY ${CGRAPH_TARGET_OBJECTS})
