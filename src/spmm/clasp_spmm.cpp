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

#include "./clasp_spmm.hpp"

#define CHECK_CUDA2(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
    }                                                                          \
}

template<class T>
Clasp_Spmm<T>::Clasp_Spmm(Dataset<T> &d, cudaDataType_t S)
:Spmm_CNN<T>(d){
    Format_cvs<T> *f = dynamic_cast<Format_cvs<T>*>(d.get_format());

    int A_nnz = f->get_A_nnz();
    int A_num_rows = f->get_A_num_rows();
    int *hA_Offsets = &(f->get_hA_Offsets()[0]);
    int vec = f->get_vec_length();
    int A_num_rows_vec = int( A_num_rows/vec );
    int hA_rows[A_num_rows_vec];

    SortedRowSwizzle(A_num_rows_vec, hA_Offsets, hA_rows);
    //IdentityRowSwizzle(A_num_rows_vec, hA_rows);

    d.get_Bmat().sync_device();
    d.get_Cmat().sync_device();
    f->reformat_and_cpy_to_device();

    CHECK_CUDA2( cudaMalloc((void**) &dA_RowIndex,
                           A_num_rows_vec * sizeof(int)) )
    CHECK_CUDA2( cudaMemcpy(dA_RowIndex, hA_rows,
                           A_num_rows_vec * sizeof(int),
                           cudaMemcpyHostToDevice) )
}

template<class T>
Clasp_Spmm<T>::~Clasp_Spmm(){
    cudaFree(dA_RowIndex);
};

template<class T>
std::vector<T>& Clasp_Spmm<T>::get_result(){
    return result;
}

template<class T>
void Clasp_Spmm<T>::SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::stable_sort(swizzle_staging.begin(), swizzle_staging.end(),
  //std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&row_offsets](int idx_a, int idx_b) {
              int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
              int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
              return length_a > length_b;
            });

  // Copy the ordered row indices to the output.
  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

template<class T>
void Clasp_Spmm<T>::IdentityRowSwizzle(int rows, int *row_indices){
    std::iota(row_indices, row_indices + rows, 0);
}

template<class T>
float Clasp_Spmm<T>::spmm(int times){
    Dataset<T>& d = this->get_dataset();
    Format_cvs<T> *f = dynamic_cast<Format_cvs<T>*>(d.get_format());

    int A_num_rows_vec = int( d.get_A_num_rows()/f->get_vec_length());
    int m=d.get_A_num_rows(), n=d.get_B_num_cols(), k=d.get_A_num_cols();
    float time;

    cudaError_t (*foo) (int m_vec, int vec_length, int k, int n,
                        const int* __restrict__ row_indices,
                        const int* __restrict__ row_offsets,
                        const int* __restrict__ column_indices,
                        const half* __restrict__ values,
                        const half* __restrict__ rhs_matrix,
                        half* __restrict__ output_matrix);
    foo =  spmm_amp::wmmaSpmm;

    CHECK_CUDA2( spmm_amp::wmmaSpmm(A_num_rows_vec,
                   f->get_vec_length(),
                   d.get_B_num_cols(),
                   d.get_A_num_cols(),
                   this->dA_RowIndex,
                   f->get_device_ptrs().csb_indptr,
                   f->get_device_ptrs().csb_indices,
                   f->get_device_ptrs().csb_values,
                   d.get_Bmat().device_ptr,
                   d.get_Cmat().device_ptr
                ) )

    time = cuTime(times, foo, A_num_rows_vec,
                   f->get_vec_length(),
                   d.get_B_num_cols(),
                   d.get_A_num_cols(),
                   this->dA_RowIndex,
                   f->get_device_ptrs().csb_indptr,
                   f->get_device_ptrs().csb_indices,
                   f->get_device_ptrs().csb_values,
                   d.get_Bmat().device_ptr,
                   d.get_Cmat().device_ptr);

    d.get_Cmat().sync_host();

    std::cout << "CLASP time: " << time << std::endl;

    std::vector<T> D_ref(m * n);
    this->result = std::vector<T>();
    for (size_t i = 0; i < m*n; i++){
        this->result.push_back(d.get_Cmat().host_ptr[i]);
    }

    return time;
}

template<>
float Clasp_Spmm<float>::spmm(int times){
    std::cerr << "only precision:half is implemented.\n";
    exit(EXIT_FAILURE);
}

template class Clasp_Spmm<float>;
template class Clasp_Spmm<__half>;