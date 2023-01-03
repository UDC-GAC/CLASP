<!--[![DOI]()]()-->
# Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs

This repo accompanies the paper [Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs](https://gac.udc.es/~basilio/papers/Castro22-MLSparse.pdf), published at PACT'22. It includes the kernels associated with CLASP, which has been presented in that conference. CLASP is a column-vector pruning-aware implementation of the SpMM routine that supports the characteristics of the Ampere platform. It aims to take advantage of the knowledge pushed into the pruning technique to generate the sparse input matrices (e.g. column-vector), and boost the performance achieved on half precision.

## Build

```
git clone --recurse-submodules git@github.com:UDC-GAC/CLASP.git
```
```
mkdir build && cd build
```
```
cmake .. -DCMAKE_BUILD_TYPE=Release -DCUDA_ARCHS="86" && make -j 12
```

Note: If you find a problem like this
```
Policy "CMP0104" is not known to this version of CMake
```
Please, comment this line ```cmake_policy(SET CMP0104 OLD)``` in ```include/sputnik/CMakeLists.txt```

## How to use
```
./src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision half --block-size 16 --m 1024 --k 256 --n 256 --d 0.2 --check
```

Results can be compared with Sputnik's library using the following notation:

```
./src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision half --m 1024 --k 256 --n 256 --d 0.2 --check
```

(Recommended before time measurement) Lock the clocks:
```
sudo nvidia-smi -i 0 -pm 1
sudo nvidia-smi -lgc 1750 -i 0
```
## Citation

```
@inproceedings{castro2022probing,
  author    = {Castro, Roberto L. and Andrade, Diego and Fraguela, Basilio B.},
  title     = {Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs},
  booktitle = {Proceedings of the International Conference on Parallel Architectures and Compilation Techniques, {PACT} 22},
  year      = {2022},
}
```


## License
Apache-2.0 License

-- Roberto López Castro
