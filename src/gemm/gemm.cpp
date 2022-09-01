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

#include "gemm.hpp"

template<class T>
Gemm<T>::Gemm(std::vector<T>& A_dense, Dataset<T> &d)
    :dset(d)
{
    A = A_dense;
    B = d.get_B();
    C = d.get_C();

    A_size = d.get_A_size();
    B_size = d.get_B_size();
    C_size = d.get_C_size();
    A_num_rows = d.get_A_num_rows();
    A_num_cols = d.get_A_num_cols();
    B_num_rows = d.get_A_num_cols();
    B_num_cols = d.get_B_num_cols();

    alpha = 1.0f;
    beta  = 0.0f;

    cudaMalloc((void**) &dA, A_size * sizeof(T));
    cudaMalloc((void**) &dB, B_size * sizeof(T));
    cudaMalloc((void**) &dC, C_size * sizeof(T));

    cudaMemcpy(dA, &A[0], A_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[0], B_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &C[0], C_size * sizeof(T),
                           cudaMemcpyHostToDevice);
}

template<class T>
Gemm<T>::Gemm(std::vector<T>& A_dense, std::vector<T>& B_dense,
                std::vector<T>& C_dense, int A_num_rows,
                int A_num_cols, int B_num_cols, Dataset<T> &d) //TODO: remove dataset object from constructor
    :dset(d)
{
    A = A_dense;
    B = B_dense;
    C = C_dense;

    A_size = A_num_rows*A_num_cols;
    B_size = A_num_cols*B_num_cols;
    C_size = A_num_rows*B_num_cols;
    this->A_num_rows = A_num_rows;
    this->A_num_cols = A_num_cols;
    this->B_num_rows = A_num_cols;
    this->B_num_cols = B_num_cols;

    alpha = 1.0f;
    beta  = 0.0f;

    cudaMalloc((void**) &dA, A_size * sizeof(T));
    cudaMalloc((void**) &dB, B_size * sizeof(T));
    cudaMalloc((void**) &dC, C_size * sizeof(T));

    cudaMemcpy(dA, &A[0], A_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dB, &B[0], B_size * sizeof(T),
                           cudaMemcpyHostToDevice);
    cudaMemcpy(dC, &C[0], C_size * sizeof(T),
                           cudaMemcpyHostToDevice);
}

template<class T>
Gemm<T>::~Gemm(){
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

template<class T>
std::vector<T>& Gemm<T>::get_C(){
    return C;
}

template class Gemm<float>;
template class Gemm<__half>;