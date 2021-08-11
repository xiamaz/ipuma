set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(UPCXX REQUIRED)
find_program(UPCXX_EXEC upcxx)
set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})

set(KLIGN_TARGET_FILES)

foreach(KMER_LENGTH ${MHM2_KMER_LENGTHS})
  set(TEMPLATE "klign")
  set(TEMPLATE_FILE "${TEMPLATE}-extern-template-${KMER_LENGTH}")
  string(TOUPPER ${TEMPLATE} TEMPLATE_UPPER)
  configure_file("${CMAKE_SOURCE_DIR}/src/extern_template.in.cpp"
                 "${TEMPLATE_FILE}.cpp" @ONLY)
  list(APPEND KLIGN_TARGET_FILES "${TEMPLATE_FILE}.cpp")
endforeach()

set(KLIGN_TARGET_OBJECTS)

foreach(tgt aligner_cpu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
endforeach()

if(ENABLE_CUDA)
  set(tgt block_align_gpu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  set(KLIGN_TARGET_OBJECTS
      ${KLIGN_TARGET_OBJECTS}
      PARENT_SCOPE)
else()
  set(tgt block_align_cpu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  set(KLIGN_TARGET_OBJECTS
      ${KLIGN_TARGET_OBJECTS}
      PARENT_SCOPE)
endif()

add_library(KLIGN_LIBRARY ${KLIGN_TARGET_FILES} ${KLIGN_TARGET_OBJECTS})

if(ENABLE_CUDA)
  include_directories(${CMAKE_CURRENT_SOURCE_DIR})
  set_property(TARGET KLIGN_LIBRARY PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  include_directories("adept-sw")
  add_subdirectory(adept-sw)
  set(MHM2_LINK_LIBRARIES ${MHM2_LINK_LIBRARIES} ADEPT_SW_LIBRARY_static)
  target_link_libraries(KLIGN_LIBRARY ${MHM2_LINK_LIBRARIES} ${UPCXX_LIBRARIES}
                        ${UPCXX_UTILS_LIBRARIES})
endif()
