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

#ifndef SPMM_H
#define SPMM_H

#include <string>
#include <map>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>

#include "../dataset/dataset.hpp"
#include "../format/format.hpp"
#include "../cuda_utils.h"
#include "../../include/sputnik/sputnik/test_utils.h"

using namespace std;

template<class T>
class Spmm_CNN {
    public:
        Spmm_CNN(Dataset<T> &d);
        Spmm_CNN(Dataset<T> &d, cudaDataType_t S);
        ~Spmm_CNN();
        T* get_dB();
        T* get_dC();
        void to_String();
        Dataset<T>& get_dataset();

        virtual float spmm(int times) = 0;
        virtual std::vector<T>& get_result() = 0;

    protected:
        Dataset<T>& dset;
        T *dB;
        T *dC;
        float gflop_count;
};

#endif