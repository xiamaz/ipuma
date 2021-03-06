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
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  endif()
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
endforeach()

if(ENABLE_CUDA)
  set(tgt block_align_gpu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  endif()
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  set(KLIGN_TARGET_OBJECTS
      ${KLIGN_TARGET_OBJECTS}
      PARENT_SCOPE)
elseif(ENABLE_POPLAR)
  set(tgt block_align_ipu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  endif()
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  set(KLIGN_TARGET_OBJECTS
      ${KLIGN_TARGET_OBJECTS}
      PARENT_SCOPE)
else()
  set(tgt block_align_cpu)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
    target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  endif()
  list(APPEND KLIGN_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
  set(KLIGN_TARGET_OBJECTS
      ${KLIGN_TARGET_OBJECTS}
      PARENT_SCOPE)
endif()

add_library(KLIGN_LIBRARY ${KLIGN_TARGET_FILES} ${KLIGN_TARGET_OBJECTS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_definitions(-DIPUMA_DEBUG)

# if(ENABLE_POPLAR)
#   add_subdirectory(ipuma-sw)
#   # add_library(IPUMA_LIBRARY "ipuma-sw/matrix.cpp" "ipuma-sw/similarity.cpp" "ipuma-sw/encoding.cpp")

#   set(MHM2_LINK_LIBRARIES ${MHM2_LINK_LIBRARIES} IPUMA_LIBRARY poplar)
# endif()

if(ENABLE_CUDA)
  set_property(TARGET KLIGN_LIBRARY PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
  include_directories("adept-sw")
  add_subdirectory(adept-sw)
  set(MHM2_LINK_LIBRARIES ${MHM2_LINK_LIBRARIES} ADEPT_SW_LIBRARY_static)
endif()

target_link_libraries(KLIGN_LIBRARY ${MHM2_LINK_LIBRARIES} ${UPCXX_LIBRARIES}
                      ${UPCXX_UTILS_LIBRARIES})
