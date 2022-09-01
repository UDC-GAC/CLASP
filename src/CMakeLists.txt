set(SPMM_BENCHS_SOURCES
	benchmark_spmm.cpp)

set_source_files_properties(${SPMM_BENCHS_SOURCES} PROPERTIES LANGUAGE CUDA)

set(SPMM_TESTS_SOURCES
	spmm_test.cpp)

set_source_files_properties(${SPMM_TESTS_SOURCES} PROPERTIES LANGUAGE CUDA)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Type of build, options are: Debug, Release" FORCE)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g2 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

add_subdirectory(format)
add_subdirectory(dataset)
add_subdirectory(gemm)
add_subdirectory(spmm)

add_executable(benchmark_spmm
	${SPMM_BENCHS_SOURCES})
set_source_files_properties(${benchmark_spmm} PROPERTIES LANGUAGE CUDA)
target_link_libraries(benchmark_spmm
	formats datasets gemm spmm boost_system boost_filesystem sputnik spmm_ve_amp ${SPUTNIK_BENCHMARK_LIBS} cudart cusparse cublas dl "/usr/local/cuda-11.2/targets/x86_64-linux/lib/libnvrtc.so")