if(NOT ENABLE_CUDA)
  message(FATAL_ERROR "Trying to build kcount-GPU but CUDA is not enabled")
endif()

set(CMAKE_CUDA_STANDARD
    14
    CACHE STRING "")
set(CMAKE_CUDA_STANDARD_REQUIRED
    ON
    CACHE BOOL "")
set(CMAKE_CUDA_EXTENSIONS
    OFF
    CACHE BOOL "")

find_package(CUDA REQUIRED)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  cmake_policy(SET CMP0074 NEW) # use the CUDA_ROOT variable
endif()

enable_language(CUDA)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  if(NOT CMAKE_CUDA_ARCHITECTURES)

    if(CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
      set(CMAKE_CUDA_ARCHITECTURES 35 50 60 70 80)
    elseif(CUDA_VERSION VERSION_GREATER_EQUAL 10.0)
      set(CMAKE_CUDA_ARCHITECTURES 30 35 50 60 70)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 30 35 50 60 70)
    endif()

    message(
      STATUS
        "Setting CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES} -- override with specific target if this build fails"
    )
  else()
    message(
      STATUS
        "Using specified CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
  endif()
else()
  find_package(CUDA REQUIRED)
  cuda_select_nvcc_arch_flags(ARCH_FLAGS_LIST)
  if(ARCH_FLAGS_LIST)
    string(REPLACE ";" " " ARCH_FLAGS "${ARCH_FLAGS_LIST}")
  else()
    foreach(_CMAKE_CUDA_COMPUTE_CAPABILITY 35 50 60 70 80)
      set(ARCH_FLAGS
          "${ARCH_FLAGS} -gencode arch=compute_${_CMAKE_CUDA_COMPUTE_CAPABILITY},code=sm_${_CMAKE_CUDA_COMPUTE_CAPABILITY}"
      )
    endforeach()
    if(NOT CMAKE_CUDA_COMPUTE_CAPABILITY)
      set(CMAKE_CUDA_COMPUTE_CAPABILITY
          70
          CACHE STRING "")
    endif()
    message(
      WARNING
        "Could not auto-detect the GPU arch flags building for all gpu compute capabilities 3 - 7 and PTX for ${CMAKE_CUDA_COMPUTE_CAPABILITY}"
    )
    set(ARCH_FLAGS
        "${ARCH_FLAGS} -gencode arch=compute_${CMAKE_CUDA_COMPUTE_CAPABILITY},code=sm_${CMAKE_CUDA_COMPUTE_CAPABILITY}"
    )
    set(ARCH_FLAGS
        "${ARCH_FLAGS} -gencode arch=compute_${CMAKE_CUDA_COMPUTE_CAPABILITY},code=compute_${CMAKE_CUDA_COMPUTE_CAPABILITY}"
    )
  endif()
  message(STATUS "ARCH_FLAGS=${ARCH_FLAGS}")
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_PTX_COMPILATION ON)

set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -Xcompiler=-Wall ${ARCH_FLAGS} -Wno-deprecated-gpu-targets"
)

message(STATUS "CMAKE_CUDA_FLAGS=${CMAKE_CUDA_FLAGS}")

add_library(KCOUNT_GPU_LIBRARY_obj OBJECT parse_and_pack.cpp gpu_hash_table.cpp)
# prime.cpp)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13)
  target_link_libraries(KCOUNT_GPU_LIBRARY_obj INTERFACE)
endif()

set_source_files_properties(parse_and_pack.cpp PROPERTIES LANGUAGE CUDA
                                                          LINKER_LANGUAGE CUDA)
set_source_files_properties(gpu_hash_table.cpp PROPERTIES LANGUAGE CUDA
                                                          LINKER_LANGUAGE CUDA)
option(KCOUNT_GPU_SHARED "kcount-GPU shared library" OFF)
option(KCOUNT_GPU_STATIC "kcount-GPU static library" ON)

if(KCOUNT_GPU_SHARED)
  add_library(KCOUNT_GPU_LIBRARY_shared SHARED
              $<TARGET_OBJECTS:KCOUNT_GPU_LIBRARY_obj>)
  set_property(TARGET KCOUNT_GPU_LIBRARY_shared
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(KCOUNT_GPU_LIBRARY_shared INTERFACE)
  install(TARGETS KCOUNT_GPU_LIBRARY_shared LIBRARY DESTINATION lib)
endif()
if(KCOUNT_GPU_STATIC)
  add_library(KCOUNT_GPU_LIBRARY_static STATIC
              $<TARGET_OBJECTS:KCOUNT_GPU_LIBRARY_obj>)
  target_link_libraries(KCOUNT_GPU_LIBRARY_static INTERFACE)
  set_property(TARGET KCOUNT_GPU_LIBRARY_static
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  install(TARGETS KCOUNT_GPU_LIBRARY_static ARCHIVE DESTINATION lib)
endif()
