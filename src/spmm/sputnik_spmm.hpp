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

#ifndef SPUTNIK_SPMM_H
#define SPUTNIK_SPMM_H

#include "spmm.hpp"

#include "../../include/sputnik/sputnik/spmm/cuda_spmm.h"

using namespace sputnik;
using namespace std;

template<class T>
class Sputnik_Spmm: public Spmm_CNN<T> {
    public:
        ~Sputnik_Spmm();
        Sputnik_Spmm(Dataset<T> &d, cudaDataType_t S);
        std::vector<T>& get_hB();
        std::vector<T>& get_hC();
        std::vector<T>& get_result();
        float spmm(int times);

    private:
        void SortedRowSwizzle(int rows, const int *row_offsets, int *row_indices);
        void IdentityRowSwizzle(int rows, int *row_indices);
        int* dA_RowIndex;
        std::vector<T> result;
};

#endif