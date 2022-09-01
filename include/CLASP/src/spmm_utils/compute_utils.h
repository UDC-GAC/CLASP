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

#ifndef SPMM_COMPUTE_UTILS_AMP_H
#define SPMM_COMPUTE_UTILS_AMP_H

namespace spmm_amp{
    template <typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct ComputeUtils {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_K / BlockWidth;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values.
        const VecType* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        __device__ __forceinline__ ComputeUtils(
            const half* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment):
            lhs_tile_(reinterpret_cast<const VecType*>(lhs_tile)),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}


        __device__ __forceinline__ void TileMAC(){
            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
                half lhs_value[VecLength];
                VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                *(lhs_value_v) = *(lhs_tile_ + n_item_idx);
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    half rhs_value = *(rhs_fragment_ + kThreadItemsK * n_item_idx + k_item_idx);
                    #pragma unroll
                    for (int v = 0; v < VecLength; v++){
                        *(output_fragment_ + k_item_idx + v * kThreadItemsK) += __half2float(lhs_value[v] * rhs_value);
                    }
                }
            }
        }

    };

    template <typename VecType, int Tile_N, int Tile_K, int BlockWidth, int VecLength>
    struct ComputeUtils1D {

        //
        // Static membrs
        //

        static constexpr int kThreadItemsK = Tile_K / BlockWidth;

        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values.
        const VecType* lhs_tile_;
        // Register file fragment storing the rhs tile
        const half* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        __device__ __forceinline__ ComputeUtils1D(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment):
            lhs_tile_(lhs_tile),
            rhs_fragment_(rhs_fragment),
            output_fragment_(output_fragment){}


        __device__ __forceinline__ void TileMAC(){
            #pragma unroll
            for (int n_item_idx = 0; n_item_idx < Tile_N; n_item_idx ++){
                half lhs_value[VecLength];
                VecType * lhs_value_v = reinterpret_cast<VecType *>(lhs_value);
                *(lhs_value_v) = *(lhs_tile_ + n_item_idx);
                #pragma unroll
                for(int k_item_idx = 0; k_item_idx < kThreadItemsK; k_item_idx ++){
                    half rhs_value = *(rhs_fragment_ + kThreadItemsK * n_item_idx + k_item_idx);
                    #pragma unroll
                    for (int v = 0; v < VecLength; v++){
                        *(output_fragment_ + k_item_idx + v * kThreadItemsK) += __half2float(lhs_value[v] * rhs_value);
                    }
                }
            }
        }

    };

    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils8 {

        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 8 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const half* lhs_tile_;
        // Register file fragment storing the rhs tile
        const float* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils8(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):

            lhs_tile_(reinterpret_cast<const half *>(lhs_tile) + lane_id * (8*2) + thread_group ),
            rhs_fragment_(reinterpret_cast<const float*>(rhs_fragment)),
            output_fragment_(output_fragment){}


        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){

            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[2];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);
            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            *(lhs_fragment) = *(lhs_tile_ + 64 * n_group_idx);
            *(lhs_fragment+1)  = *(lhs_tile_ + 8 + 64 * n_group_idx );
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[0] = rhs_fragment_half[i*2];
                rhs_fragment_h[1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[3] = rhs_fragment_half[i*2+8+1];

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[0]), "r"(rhs_fragment_int[1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[2];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);
            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            *(lhs_fragment) = *(lhs_tile_ + 64 * n_group_idx);
            *(lhs_fragment+1)  = *(lhs_tile_ + 8 + 64 * n_group_idx );
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[0] = rhs_fragment_half[i*2];
                rhs_fragment_h[1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[3] = rhs_fragment_half[i*2+8+1];

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[0]), "r"(rhs_fragment_int[1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }
    };


    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils4 {
        //
        // Static members
        //

        static constexpr int kTotalStep = Tile_N / 8 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const half* lhs_tile_;
        // Register file fragment storing the rhs tile
        const float* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils4(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):

            lhs_tile_(reinterpret_cast<const half *>(lhs_tile) + lane_id * (4*2) + thread_group ),
            rhs_fragment_(reinterpret_cast<const float*>(rhs_fragment)),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}

        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[8];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);

            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            if (thread_group_ < 4){
                *(lhs_fragment) = *(lhs_tile_ + 32 * n_group_idx); //32=4*8
                *(lhs_fragment+1)  = *(lhs_tile_ + 4 + 32 * n_group_idx ); //4=vecLength
            }
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[i*4] = rhs_fragment_half[i*2];
                rhs_fragment_h[i*4+1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[i*4+2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[i*4+3] = rhs_fragment_half[i*2+8+1];

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[i*2]), "r"(rhs_fragment_int[i*2+1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[8];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);

            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            if (thread_group_ < 4){
                *(lhs_fragment) = *(lhs_tile_ + 32 * n_group_idx); //32=4*8
                *(lhs_fragment+1)  = *(lhs_tile_ + 4 + 32 * n_group_idx ); //4=vecLength
            }
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[i*4] = rhs_fragment_half[i*2];
                rhs_fragment_h[i*4+1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[i*4+2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[i*4+3] = rhs_fragment_half[i*2+8+1];

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[i*2]), "r"(rhs_fragment_int[i*2+1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }
    };

    // Compute Tile for k=2
    template <typename VecType, int Tile_N>
    struct wmmaComputeUtils2 {
        //
        // Static members
        //
        static constexpr int kTotalStep = Tile_N / 8 - 1;
        //
        // Member variables
        //

        // Shared memory buffer storing the lhs tile values
        const half* lhs_tile_;
        // Register file fragment storing the rhs tile
        const float* rhs_fragment_;
        // Register file fragment to accumulate results into.
        float* output_fragment_;
        int thread_group_;

        // Constructor
        __device__ __forceinline__ wmmaComputeUtils2(
            const VecType* lhs_tile,
            const half* rhs_fragment,
            float* output_fragment,
            int lane_id, int thread_group):

            lhs_tile_(reinterpret_cast<const half *>(lhs_tile) + lane_id * (2*2) + thread_group ),
            rhs_fragment_(reinterpret_cast<const float*>(rhs_fragment)),
            output_fragment_(output_fragment),
            thread_group_(thread_group){}

        // Compute
        __device__ __forceinline__ void TileMAC(int n_group_idx){
            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[2];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);
            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[0] = rhs_fragment_half[i*2];
                rhs_fragment_h[1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[3] = rhs_fragment_half[i*2+8+1];
            }
            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            if (thread_group_ < 2){
                *(lhs_fragment) = *(lhs_tile_ + 16 * n_group_idx); //32=4*8
                *(lhs_fragment+1)  = *(lhs_tile_ + 2 + 16 * n_group_idx ); //4=vecLength
            }
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[0]), "r"(rhs_fragment_int[1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }

        // Compute Residue
        __device__ __forceinline__ void TileMACResidue(int n_group_idx){
            const half *rhs_fragment_half = reinterpret_cast<const half*>(rhs_fragment_ + 8 * n_group_idx);
            unsigned rhs_fragment_aux[2];
            half* rhs_fragment_h = reinterpret_cast<half*>(rhs_fragment_aux);

            #pragma unroll
            for (int i = 0; i < 4; i++){
                rhs_fragment_h[0] = rhs_fragment_half[i*2];
                rhs_fragment_h[1] = rhs_fragment_half[i*2+8];
                rhs_fragment_h[2] = rhs_fragment_half[i*2+1];
                rhs_fragment_h[3] = rhs_fragment_half[i*2+8+1];
            }

            const unsigned *rhs_fragment_int = reinterpret_cast<unsigned*>(rhs_fragment_aux);

            __half lhs_fragment[2];
            if (thread_group_ < 2){
                *(lhs_fragment) = *(lhs_tile_ + 16 * n_group_idx); //32=4*8
                *(lhs_fragment+1)  = *(lhs_tile_ + 2 + 16 * n_group_idx ); //4=vecLength
            }
            const unsigned *lhs_fragment_int = reinterpret_cast<unsigned*>(lhs_fragment);

            #pragma unroll
            for (int i = 0; i < 4; i++){

                asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 \t"
                    "{%0, %1, %2, %3}, \t"
                    "{%4, %5}, \t"
                    "{%6}, \t"
                    "{%0, %1, %2, %3}; ":
                    "+f"(output_fragment_[0 + 2 * i]), "+f"(output_fragment_[8+ 0 + 2 * i]),
                    "+f"(output_fragment_[1 + 2 * i]), "+f"(output_fragment_[8+ 1 + 2 * i]):
                    "r"(rhs_fragment_int[0]), "r"(rhs_fragment_int[1]),
                    "r"(lhs_fragment_int[0])
                );
            }
        }
    };
}

#endif