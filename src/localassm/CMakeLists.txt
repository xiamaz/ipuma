# use c++14 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(UPCXX REQUIRED)
find_program(UPCXX_EXEC upcxx)
set(CMAKE_CXX_COMPILER ${UPCXX_EXEC})

set(LOCALASSM_TARGET_OBJECTS)

foreach(tgt localassm localassm_core)
  add_library(${tgt} OBJECT ${tgt}.cpp)
  target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
  list(APPEND LOCALASSM_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
endforeach()


if(ENABLE_CUDA)
    foreach(tgt localassm_gpu)
        add_library(${tgt} OBJECT ${tgt}.cpp)
        target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
        list(APPEND LOCALASSM_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
        set(LOCALASSM_TARGET_OBJECTS ${LOCALASSM_TARGET_OBJECTS} PARENT_SCOPE)
    endforeach()
else()
    foreach(tgt localassm_cpu)
        add_library(${tgt} OBJECT ${tgt}.cpp)
        target_link_libraries(${tgt} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
        list(APPEND LOCALASSM_TARGET_OBJECTS $<TARGET_OBJECTS:${tgt}>)
        set(LOCALASSM_TARGET_OBJECTS ${LOCALASSM_TARGET_OBJECTS} PARENT_SCOPE)
    endforeach()
endif()

add_library(LOCALASSM_LIBRARY ${LOCALASSM_TARGET_OBJECTS})

if(ENABLE_CUDA)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    set_property(TARGET LOCALASSM_LIBRARY PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
    include_directories("localassm-gpu")
    add_subdirectory(localassm-gpu)
    set(MHM2_LINK_LIBRARIES ${MHM2_LINK_LIBRARIES} LOCALASSM_GPU_LIBRARY_static)
    target_link_libraries(LOCALASSM_LIBRARY ${MHM2_LINK_LIBRARIES} ${UPCXX_LIBRARIES} ${UPCXX_UTILS_LIBRARIES})
endif()
