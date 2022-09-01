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

#ifndef SPMM_OUTPUT_Tile_AMP_H
#define SPMM_OUTPUT_Tile_AMP_H

struct __align__(16) half8 {
  half2 x, y, z, w;
};

namespace spmm_amp{
    template <typename LoadType, typename OutType, int Tile_K, int BlockWidth, int VecLength>
    struct OutputTile{

        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(LoadType) / sizeof(OutType);
        static constexpr int kValuesPerLoad = sizeof(LoadType) / sizeof(half);
        static constexpr int kThreadItemsK_ = Tile_K / BlockWidth / kValuesPerStore_;
        static constexpr int kScaler_ = sizeof(OutType)/sizeof(half);

        //
        // Member variables
        //

        // The register file fragment with the results to store
        const LoadType* output_fragment_;
        // The output matrix pointer in global memory
        LoadType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ OutputTile(
            int row_offset_vec, int column_offset,
            int cols, int thread_idx_x,
            const float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<const LoadType *>(output_fragment);
            const int output_offset = row_offset_vec * VecLength * cols + column_offset;
            output_matrix_ = reinterpret_cast<LoadType *>(output_matrix + output_offset) + thread_idx_x * kScaler_;
            rhs_columns_ = cols / kValuesPerStore_ * kScaler_;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if (kValuesPerLoad == kValuesPerStore_){
                #pragma unroll
                for (int v = 0; v < VecLength; v++){
                    const LoadType * output_fragment_t = output_fragment_ + v * 2 * kThreadItemsK_;
                    LoadType * output_matrix_t = output_matrix_ + v * rhs_columns_;
                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                        float values [kValuesPerStore_];
                        LoadType * values_loadType = reinterpret_cast<LoadType *>(values);
                        OutType *values_outType = reinterpret_cast<OutType *>(values);
                        *(values_loadType) = *(output_fragment_t);
                        *(values_loadType + 1) = *(output_fragment_t + 1);
                        #pragma unroll
                        for (int dv = 0; dv < kValuesPerStore_; dv ++){
                            values_outType[dv] = (OutType)values[dv];
                        }
                        *(output_matrix_t) = *(values_loadType);
                        output_fragment_t += 2;
                        output_matrix_t +=BlockWidth;
                    }
                }
            }
            else{
                #pragma unroll
                for (int v = 0; v < VecLength; v++){
                    const LoadType * output_fragment_t = output_fragment_ + v * 2 * kThreadItemsK_;
                    LoadType * output_matrix_t = output_matrix_ + v * rhs_columns_;
                    #pragma unroll
                    for (int k_item_idx = 0; k_item_idx < kThreadItemsK_; k_item_idx ++){
                        *(output_matrix_t) = *(output_fragment_t);
                        *(output_matrix_t + 1) = *(output_fragment_t + 1);
                        output_matrix_t +=BlockWidth * 2;
                        output_fragment_t += 2;
                    }
                }
            }
        }
    };

    template<typename OutType, typename StoreType>
    struct wmmaOutputTile8{
        //
        // Static members
        //

        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile8(
            int lane_id, int thread_group,
            int row_offset_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            const int output_offset = (row_offset_vec * 8 + lane_id*2) * cols + column_offset + (thread_group) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }


            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            *(output_matrix_) = *(output_fragment_storetype);
            *(output_matrix_ + rhs_columns_) = *(output_fragment_storetype + 1);

        }
    };
    template<typename OutType, typename StoreType>
    struct wmmaOutputTile4{
        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile4(
            int lane_id, int thread_group,
            int row_offset_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            const int output_offset = (row_offset_vec * 4 + lane_id*2) * cols + column_offset + (thread_group) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }


            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            if(lane_id_<2){
                *(output_matrix_) = *(output_fragment_storetype);
                *(output_matrix_ + rhs_columns_) = *(output_fragment_storetype + 1);
            }

        }
    };

    template<typename OutType, typename StoreType>
    struct wmmaOutputTile2{
        static constexpr int kValuesPerStore_ = sizeof(StoreType) / sizeof(OutType);
        static constexpr int kTypeConvert = sizeof(OutType) / sizeof(float);
        // static constexpr int kValuesPerStore_ = sizeof(float4) / sizeof(float);

        //
        // Member variables
        //
        int lane_id_;
        int thread_group_;
        // The register file fragment with the results to store
        float2* output_fragment_;
        StoreType* output_matrix_;
        // The number of columns in the rhs matrix
        int rhs_columns_;

        // Constructor
        __device__ __forceinline__ wmmaOutputTile2(
            int lane_id, int thread_group,
            int row_offset_vec, int column_offset,
            int cols,
            float* output_fragment,
            OutType* output_matrix)
        {
            output_fragment_ = reinterpret_cast<float2 *>(output_fragment);
            const int output_offset = (row_offset_vec * 2 + lane_id*2) * cols + column_offset + (thread_group) * 8;
            output_matrix_ = reinterpret_cast<StoreType *>(output_matrix + output_offset);
            rhs_columns_ = cols / kValuesPerStore_;
            lane_id_ = lane_id;
            thread_group_ = thread_group;
        }

        // Store
        __device__ __forceinline__ void Store(){
            if (kTypeConvert != 1){
                float* output_fragment_float = reinterpret_cast<float *>(output_fragment_);
                OutType* output_fragment_outType = reinterpret_cast<OutType *>(output_fragment_);
                #pragma unroll
                for(int i = 0; i < 16; i++){
                    output_fragment_outType[i] = (OutType)output_fragment_float[i];
                }
            }


            StoreType *output_fragment_storetype = reinterpret_cast<StoreType *>(output_fragment_);
            if(lane_id_<1){
                *(output_matrix_) = *(output_fragment_storetype);
                *(output_matrix_ + rhs_columns_) = *(output_fragment_storetype + 1);
            }

        }
    };
}
#endif