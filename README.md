<!--[![DOI]()]()-->
# Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs

PACT 2022 paper entitled "Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs"

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
Roberto L. Castro, Diego Andrade, and Basilio B. Fraguela. 2022. Probing the Efficacy of Hardware-Aware Weight Pruning to Optimize the SpMM routine on Ampere GPUs. In PACT ’22: International Conference on Parallel Architectures and Compilation Techniques (PACT), October 10–12, 2022, Chicago, IL.
```
## License
Apache-2.0 License

-- Roberto López Castro
