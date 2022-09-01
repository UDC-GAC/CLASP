set(SPMM_VE_AMP_SOURCES
    ../include/cublas_gemm.cuh
	cublas_gemm.cu
	../include/cuda_spmm.cuh
    cuda_spmm.cu
	../include/wmma_spmm.cuh
	wmma_spmm.cu
)

set_source_files_properties(${SPMM_VE_AMP_SOURCES} PROPERTIES LANGUAGE CUDA)

#add_library(spmm SHARED
#	${SPMM_SOURCES})

add_library(spmm_ve_amp STATIC
	${SPMM_VE_AMP_SOURCES})

target_link_libraries(spmm_ve_amp
    PUBLIC cudart cusparse cublas nvToolsExt
)

target_include_directories(spmm_ve_amp
	PUBLIC .)