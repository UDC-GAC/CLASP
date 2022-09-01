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

#include "cublas_gemm.hpp"

template<class T>
Cublas_gemm<T>::Cublas_gemm(std::vector<T>& A_dense, Dataset<T> &d)
    :Gemm<T>(A_dense, d)
{
    cublasCreate(&handle);
}

template<class T>
Cublas_gemm<T>::~Cublas_gemm(){
    cublasDestroy(handle);
}

template<>
inline float Cublas_gemm<float>::sgemm(int times, int num_batches){
    float time;
    // Performs warmup operation
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->B_num_cols, this->A_num_rows, this->B_num_rows,
                &(this->alpha),
                this->dB, this->B_num_cols,
                this->dA, this->B_num_rows,
                &(this->beta),
                this->dC, this->B_num_cols
                );

    ///////////////////////
    time = cuTime(times, cublasSgemm, handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->B_num_cols, this->A_num_rows, this->B_num_rows,
                &(this->alpha),
                this->dB, this->B_num_cols,
                this->dA, this->B_num_rows,
                &(this->beta),
                this->dC, this->B_num_cols);

    std::cout << "cuBlas time: " << time << std::endl;

    float *hC = &(this->get_C()[0]);
    cudaMemcpy(hC, this->dC, this->dset.get_C_size() * sizeof(float),
                           cudaMemcpyDeviceToHost);

    return time;
}

template<>
inline float Cublas_gemm<half>::sgemm(int times, int num_batches){
    float time;
    // Performs warmup operation

    cublasHgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->B_num_cols, this->A_num_rows, this->B_num_rows,
                &(this->alpha),
                this->dB, this->B_num_cols,
                this->dA, this->B_num_rows,
                &(this->beta),
                this->dC, this->B_num_cols
                );

    ///////////////////////
    time = cuTime(times, cublasHgemm, handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                this->B_num_cols, this->A_num_rows, this->B_num_rows,
                &(this->alpha),
                this->dB, this->B_num_cols,
                this->dA, this->B_num_rows,
                &(this->beta),
                this->dC, this->B_num_cols);

    std::cout << "cuBlas time: " << time << std::endl;

    half *hC = &(this->get_C()[0]);
    cudaMemcpy(hC, this->dC, this->dset.get_C_size() * sizeof(half),
                           cudaMemcpyDeviceToHost);

    return time;
}

template class Cublas_gemm<float>;
template<> float Cublas_gemm<float>::sgemm(int times, int num_batches);
template class Cublas_gemm<half>;
template<> float Cublas_gemm<half>::sgemm(int times, int num_batches);