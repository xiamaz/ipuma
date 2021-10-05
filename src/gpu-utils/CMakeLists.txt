if(NOT ENABLE_CUDA)
  message(FATAL_ERROR "Trying to build GPU-UTILS but CUDA is not enabled")
endif()

add_library(GPU_UTILS_LIBRARY_obj OBJECT gpu_utils.cpp gpu_common.cpp)
if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.13)
  target_link_libraries(GPU_UTILS_LIBRARY_obj INTERFACE)
endif()

set_source_files_properties(gpu_utils.cpp gpu_common.cpp
                            PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
option(GPU_UTILS_SHARED "GPU-utils shared library" OFF)
option(GPU_UTILS_STATIC "GPU-utils static library" ON)

if(GPU_UTILS_SHARED)
  add_library(GPU_UTILS_LIBRARY_shared SHARED
              $<TARGET_OBJECTS:GPU_UTILS_LIBRARY_obj>)
  set_property(TARGET GPU_UTILS_LIBRARY_shared
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(GPU_UTILS_LIBRARY_shared INTERFACE)
  install(TARGETS GPU_UTILS_LIBRARY_shared LIBRARY DESTINATION lib)
endif()
if(GPU_UTILS_STATIC)
  add_library(GPU_UTILS_LIBRARY_static STATIC
              $<TARGET_OBJECTS:GPU_UTILS_LIBRARY_obj>)
  target_link_libraries(GPU_UTILS_LIBRARY_static INTERFACE)
  set_property(TARGET GPU_UTILS_LIBRARY_static
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  install(TARGETS GPU_UTILS_LIBRARY_static ARCHIVE DESTINATION lib)
endif()
