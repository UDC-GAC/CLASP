cmake_minimum_required(VERSION 3.13)

project(VECTORSPARSEAMPERE CUDA CXX C)

##
### Handle Dependencies.
##
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

##
### C++ and CUDA flags.
##

# Requires C++11.
set(CMAKE_CXX_STANDARD 11)

#cmake_policy(SET CMP0104 OLD)

if(BUILD_TEST)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPUTNIK_BUILD_TEST")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DSPUTNIK_BUILD_TEST")
endif()

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

#add_subdirectory(projects/sputnik)

##
### Project build.
##

set(SPMM_VE_AMP_BENCHS_SOURCES
	spmm_benchmark.cpp)

set_source_files_properties(${SPMM_VE_AMP_BENCHS_SOURCES} PROPERTIES LANGUAGE CUDA)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Type of build, options are: Debug, Release" FORCE)
endif()

# set compile options
#add_compile_options(-Wall -Wextra)
#set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_FLAGS_DEBUG "-g2 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(VECTORSPARSE_AMP_SRCS)
add_subdirectory(src)

##################################

add_executable(spmm_benchmark_ve_amp
	${SPMM_VE_AMP_BENCHS_SOURCES})
set_source_files_properties(${spmm_benchmark_ve_amp} PROPERTIES LANGUAGE CUDA)  	
target_link_libraries(spmm_benchmark_ve_amp
	spmm_ve_amp boost_system boost_filesystem sputnik ${SPUTNIK_BENCHMARK_LIBS})

#string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE_UCASE)
#message(STATUS ">> CXX Compile Flags (For ${CMAKE_BUILD_TYPE}):\
# ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UCASE}}")

#set(INSTALL_BASE "include/vectorSparse")
#install(FILES "src/cuda_spmm.h" DESTINATION "${INSTALL_BASE}/spmm")