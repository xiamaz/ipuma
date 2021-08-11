# use c++14 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(UPCXX REQUIRED)
find_program(UPCXX_EXEC upcxx)
set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})

set(KCOUNT_TARGET_FILES)

foreach(KMER_LENGTH ${MHM2_KMER_LENGTHS})
  foreach(TEMPLATE "kcount")
    set(TEMPLATE_FILE "${TEMPLATE}-extern-template-${KMER_LENGTH}")
    string(TOUPPER ${TEMPLATE} TEMPLATE_UPPER)
    configure_file("${CMAKE_SOURCE_DIR}/src/extern_template.in.cpp" "${TEMPLATE_FILE}.cpp" @ONLY)
    list(APPEND KCOUNT_TARGET_FILES "${TEMPLATE_FILE}.cpp")
  endforeach()
endforeach()

add_library(KCOUNT_LIBRARY ${KCOUNT_TARGET_FILES})

if(ENABLE_CUDA)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET KCOUNT_LIBRARY PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF) # just to be sure. It should be off in parent dir
    include_directories("kcount-gpu")
    add_subdirectory(kcount-gpu)
    set(MHM2_LINK_LIBRARIES ${MHM2_LINK_LIBRARIES} KCOUNT_GPU_LIBRARY_static)
    target_link_libraries(KCOUNT_LIBRARY ${MHM2_LINK_LIBRARIES} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
endif()

