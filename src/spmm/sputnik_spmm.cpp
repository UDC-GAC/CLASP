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

#include "./sputnik_spmm.hpp"

template<class T>
Sputnik_Spmm<T>::Sputnik_Spmm(Dataset<T> &d, cudaDataType_t S)
:Spmm_CNN<T>(d){
    Format_csr<T> *f = dynamic_cast<Format_csr<T>*>(d.get_format());

    int A_nnz = f->get_A_nnz();
    int A_num_rows = f->get_A_num_rows();
    int *hA_Offsets = &(f->get_hA_Offsets()[0]);
    int hA_rows[A_num_rows];

    SortedRowSwizzle(A_num_rows, hA_Offsets, hA_rows);

    d.get_Bmat().sync_device();
    d.get_Cmat().sync_device();
    f->reformat_and_cpy_to_device();

    cudaMalloc((void**) &dA_RowIndex, A_num_rows * sizeof(int));
    cudaMemcpy(dA_RowIndex, hA_rows, A_num_rows * sizeof(int), cudaMemcpyHostToDevice);
}

template<class T>
void Sputnik_Spmm<T>::SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices) {
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
void Sputnik_Spmm<T>::IdentityRowSwizzle(int rows, int *row_indices){
    std::iota(row_indices, row_indices + rows, 0);
}


template<class T>
Sputnik_Spmm<T>::~Sputnik_Spmm(){
  cudaFree(dA_RowIndex);
}


template<class T>
std::vector<T>& Sputnik_Spmm<T>::get_result(){
    return result;
}

template<>
inline float Sputnik_Spmm<float>::spmm(int times){
    Dataset<float>& d = this->get_dataset();
    Format_csr<float> *f = dynamic_cast<Format_csr<float>*>(d.get_format());
    cudaStream_t stream = 0;

    int A_nnz = f->get_A_nnz();
    int m=d.get_A_num_rows(), n=d.get_B_num_cols(), k=d.get_A_num_cols();

    // Performs warmup operation
    CUDA_CALL(CudaSpmm(
          m, k, n,
          A_nnz,
          this->dA_RowIndex, f->get_device_ptrs().csb_values,
          f->get_device_ptrs().csb_indptr, f->get_device_ptrs().csb_indices,
          d.get_Bmat().device_ptr, d.get_Cmat().device_ptr, 0));

    float time;

    cudaError_t (*foo)(int m, int k, int n, int nonzeros,
                     const int* __restrict__ row_indices,
                     const float* __restrict__ values,
                     const int* __restrict__ row_offsets,
                     const int* __restrict__ column_indices,
                     const float* __restrict__ dense_matrix,
                     float* __restrict__ output_matrix,
                     cudaStream_t stream);
    foo = CudaSpmm;

    time = cuTime(times, foo, m, k, n,
          A_nnz,
          this->dA_RowIndex, f->get_device_ptrs().csb_values,
          f->get_device_ptrs().csb_indptr, f->get_device_ptrs().csb_indices,
          d.get_Bmat().device_ptr, d.get_Cmat().device_ptr, stream);

    d.get_Cmat().sync_host();

    std::cout << "Sputnik time: " << time << std::endl;

    std::vector<float> D_ref(m * n);
    this->result = std::vector<float>();
    for (size_t i = 0; i < m*n; i++){
        this->result.push_back(d.get_Cmat().host_ptr[i]);
    }

    return time;
}

template<>
inline float Sputnik_Spmm<half>::spmm(int times){
    Dataset<half>& d = this->get_dataset();
    Format_csr<half> *f = dynamic_cast<Format_csr<half>*>(d.get_format());
    cudaStream_t stream = 0;

    int A_nnz = f->get_A_nnz();
    int m=d.get_A_num_rows(), n=d.get_B_num_cols(), k=d.get_A_num_cols();

    std::vector<short> hA_columns_short;
    for(auto i=0; i<A_nnz; i++){
        hA_columns_short.push_back(static_cast<short>(f->get_hA_col()[i]));
    }
    short *hA_columns = &(hA_columns_short[0]);
    int* dA_columns;
    cudaMalloc((void**) &dA_columns, A_nnz * sizeof(short));
    cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(short), cudaMemcpyHostToDevice);

    const half2 *values = reinterpret_cast<const half2*>(f->get_device_ptrs().csb_values);
    const half2 *B_2 = reinterpret_cast<const half2*>(d.get_Bmat().device_ptr);
    half2 *C_2 = reinterpret_cast<half2*>(d.get_Cmat().device_ptr);
    const short2 *columns = reinterpret_cast<const short2*>(dA_columns);
    const int* dA_RowIndex_ = reinterpret_cast<const int*>(dA_RowIndex);
    const int* dA_csrOffsets_= reinterpret_cast<const int*>(f->get_device_ptrs().csb_indptr);


    cudaError_t (*foo) (int m, int k, int n, int nonzeros,
                     const int* __restrict__ row_indices,
                     const half2* __restrict__ values,
                     const int* __restrict__ row_offsets,
                     const short2* __restrict__ column_indices,
                     const half2* __restrict__ dense_matrix,
                     half2* __restrict__ output_matrix,
                     cudaStream_t stream);
    foo = CudaSpmm;

    // Performs warmup operation
    CUDA_CALL(CudaSpmm(m, k, n, A_nnz,
                       dA_RowIndex_, values,
                        dA_csrOffsets_, columns,
                        B_2, C_2, 0));

    float time;

    time = cuTime(times, foo, m, k, n, A_nnz,
                       dA_RowIndex_, values,
                        dA_csrOffsets_, columns,
                        B_2, C_2, stream);

    d.get_Cmat().sync_host();

    std::cout << "Sputnik time: " << time << std::endl;

    std::vector<half> D_ref(m * n);
    this->result = std::vector<half>();
    for (size_t i = 0; i < m*n; i++){
        this->result.push_back(d.get_Cmat().host_ptr[i]);
    }

    return time;
}

template class Sputnik_Spmm<float>;
template<> float Sputnik_Spmm<float>::spmm(int times);
template class Sputnik_Spmm<__half>;
template<> float Sputnik_Spmm<__half>::spmm(int times);