/*
 * Copyright (C) 2022 Roberto Lopez Castro (roberto.lopez.castro@udc.es)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>
#include <cuda_fp16.h>

#include "./format/format_cxx.hpp"
#include "./format/format_csr.hpp"
#include "./format/format_cvs.hpp"

#include "./dataset/dataset.hpp"

#include "./spmm/sputnik_spmm.hpp"
#include "./spmm/clasp_spmm.hpp"

#include "./gemm/cublas_gemm.hpp"

#include "./util/argparse.h"

#define TIMES 50

using namespace std;

template<typename T>
int check_results(std::vector<T> cusparse_result, std::vector<T> cuBLAS_result, Dataset<T>& d, int num_batches){
    int errors = 0;
    for(int i=0; i<num_batches; i++){
        for (int j=0; j < d.get_A_num_rows() * d.get_B_num_cols() ; j++){
            float c_value  = static_cast<float>(cusparse_result[i*(d.get_C_size()+128)+ j]);
            float c_result = static_cast<float>(cuBLAS_result  [i*d.get_C_size()+ j]);
            if (abs(c_value - c_result) > 1.0){
                std::cout << "(" << i << "," << j << "): " << c_value << " vs " << c_result << std::endl;
                errors ++;
                break;
            }
        }
    }

    if (errors > 0){
        printf("spmm_example test FAILED: wrong result\n");
        return 1;
    } else {
        printf("spmm_example test PASSED\n");
        return 0;
    }

    return errors;
}

template<typename T>
Dataset<T>* create_dataset(int m, int n, int k, float density, Format<T> &fm, int block_sz, int seed){
    Dataset<T> *dset;

    dset = new Dataset<T>(m, k, n, density, fm, seed, block_sz);

    return dset;
}

template<typename T>
Format<T>* create_sparse_format(int pattern_code){
    Format<T> *fm;
    switch (pattern_code)
    {
    case 0:
        fm = new Format_csr<T>();
        break;
    case 1:
        fm = new Format_cvs<T>();
        break;
    default:
        break;
    }

    return fm;
}

template<typename T>
Gemm<T>* create_gemm(int gemm_code, std::vector<T>& A_dense, Dataset<T> &d){
    Gemm<T>* gemm;

    switch (gemm_code)
    {
    case 0:
        gemm = new Cublas_gemm<T>(A_dense, d);
        break;

    default:
        break;
    }

    return gemm;
}

template<typename T>
Spmm_CNN<T>* create_spmm(int spmm_code, Dataset<T> &d, cudaDataType_t S){
    Spmm_CNN<T>* spmm;

    switch (spmm_code)
    {
    case 0:
        spmm = new Clasp_Spmm<T>(d, S);
        break;
    case 1:
        spmm = new Sputnik_Spmm<T>(d, S);
        break;

    default:
        break;
    }

    return spmm;
}

template<typename T>
void launch_kernels(int pattern_code, int gemm_code, int spmm_code, int check, int m, int n, int k, float density, int block_sz, int seed, cudaDataType_t S){
    Format<T> *fm = create_sparse_format<T>(pattern_code);
    Dataset<T> *dt = create_dataset<T>(m, n, k, density, *fm, block_sz, seed);

    Gemm<T> *gemm = create_gemm(gemm_code, fm->to_dense(), *dt);

    if(block_sz>8 && typeid(*fm)==typeid(Format_cvs<T>)){
        Format_cvs<T> *f_tmp = dynamic_cast<Format_cvs<T>*>(dt->get_format());
        f_tmp->change_v_length(8);
    }

    Spmm_CNN<T> *spmm = create_spmm(spmm_code, *dt, S);

    cudaProfilerStart();
    spmm->spmm(TIMES);
    gemm->sgemm(TIMES);
    cudaProfilerStop();

    if(check)
        check_results(gemm->get_C(), spmm->get_result(), *dt, 1);

    delete fm;
    delete dt;
    delete gemm;
    delete spmm;
}

int main(int argc, const char **argv) {
    int m, n, k;
    int pattern_code, block_sz;
    float density;
    unsigned seed;
    int spmm_code, gemm_code, precision_code;
    bool check;

    parseArgs(argc, argv, m, n, k, density, block_sz, spmm_code, gemm_code, pattern_code, precision_code, seed, check, /* verbose */false);

    if(precision_code) {
        launch_kernels<half>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, seed, CUDA_R_16F);
    } else {
        launch_kernels<float>(pattern_code, gemm_code, spmm_code, check, m, n, k, density, block_sz, seed, CUDA_R_32F);
    }

}