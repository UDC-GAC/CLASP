set(SPMM_SOURCES
	spmm.hpp
	spmm.cpp
	sputnik_spmm.hpp
	sputnik_spmm.cpp
	clasp_spmm.hpp
	clasp_spmm.cpp
)

set_source_files_properties(${SPMM_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(spmm STATIC
	${SPMM_SOURCES})

target_link_libraries(spmm
    PUBLIC cudart cusparse cublas nvToolsExt cusparseLt dl "/usr/local/cuda-11.2/targets/x86_64-linux/lib/libnvrtc.so"
)

target_include_directories(spmm
	PUBLIC .)