if(NOT ENABLE_CUDA)
  message(FATAL_ERROR "Trying to build ADEPT-SW but CUDA is not enabled")
endif()

add_library(GPU_LOC_obj OBJECT kernel.cpp driver.cpp)

set_source_files_properties(kernel.cpp driver.cpp
                            PROPERTIES LANGUAGE CUDA LINKER_LANGUAGE CUDA)
option(LOCALASSM_GPU_LIBRARY_shared "LOCALASSM_GPU shared library" OFF)
option(LOCALASSM_GPU_LIBRARY_static "LOCALASSM_GPU static library" ON)

if(LOCALASSM_GPU_LIBRARY_shared)
  add_library(LOCALASSM_GPU_LIBRARY_shared SHARED $<TARGET_OBJECTS:GPU_LOC_obj>)
  set_property(TARGET LOCALASSM_GPU_LIBRARY_shared
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  install(TARGETS LOCALASSM_GPU_LIBRARY_shared LIBRARY DESTINATION lib)
endif()
if(LOCALASSM_GPU_LIBRARY_static)
  add_library(LOCALASSM_GPU_LIBRARY_static $<TARGET_OBJECTS:GPU_LOC_obj>)
  set_property(TARGET LOCALASSM_GPU_LIBRARY_static
               PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  install(TARGETS LOCALASSM_GPU_LIBRARY_static ARCHIVE DESTINATION lib)
endif()
