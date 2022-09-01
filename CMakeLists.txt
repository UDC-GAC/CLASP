cmake_minimum_required(VERSION 3.13)

project(SpMM CUDA CXX C)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

set(CMAKE_CXX_STANDARD           17)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")
add_definitions(-DGPU_CC=${CUDA_ARCHS})

include_directories(include/sputnik)
add_subdirectory(include/sputnik)
add_subdirectory(include/CLASP)
add_subdirectory(src)