import os.path
import os
import pandas as pd
import sys

dimensions = [
[64   , 64  , 3136],
[ 512 , 4608  ,  49],
[ 256 , 1024  , 196],
[1024 ,  512  , 196],
[ 256 , 1024  , 196],
[ 256 ,   64  ,3136],
[ 128 ,  512  , 784],
[ 256 , 1024  , 196],
[1024 ,  256  , 196],
[ 256 , 2304  , 196],
[  64 ,  147  ,2544],
[1024 ,  256  , 196],
[ 128 , 1152  , 784],
[ 128 , 1152  , 784],
[ 256 , 2304  , 196],
[ 512 ,  128  , 784],
[ 256 ,   64  ,3136],
[ 128 , 1152  , 784],
[ 256 , 1024  , 196],
[2048 ,  512  ,  49],
[ 128 ,  512  , 784],
[ 256 , 2304  , 196],
[  64 ,  576  ,3136],
[2048 , 1024  ,  49],
[2048 ,  512  ,  49],
[ 256 , 2304  , 196],
[  64 ,  576  ,3136],
[ 512 ,  128  , 784],
[  64 ,  256  ,3136],
[ 256 ,  512  , 784],
[ 128 ,  512  , 784],
[1024 ,  256  , 196],
[1024 ,  256  , 196],
[ 256 , 1024  , 196],
[ 512 , 2048  ,  49],
[ 128 , 1152  , 784],
[ 512 , 4608  ,  49],
[  64 ,  256  ,3136],
[ 512 ,  256  , 784],
[ 128 ,  256  ,3136],
[ 256 , 2304  , 196],
[ 256 ,   64  ,3136],
[ 512 ,  128  , 784],
[ 512 ,  128  , 784],
[  64 ,  576  ,3136],
[ 256 , 2304  , 196],
[ 512 , 1024  , 196],
[ 512 , 4608  ,  49],
[1024 ,  256  , 196],
[1024 ,  256  , 196],
[ 256 ,   64  ,3136],
[ 512 , 2048  ,  49],
[2048 ,  512  ,  49]
]

dim3 =  [[2048,512,256]]*12 + [[512,2048,256]]*12 + [[512,512,256]]*72

from math import log10, floor
def round_to_1(x):
    return round(x, -int(floor(log10(abs(x)))))

def bench(lib, name, m, k, n, v, precision, wm="8"):
    file = name+"_"+precision+"_"+m+"_"+k+"_"+n+"_v"+v
    path = "tmp2"+file

    for gp in [50, 70, 80, 85, 87, 90, 95, 98]:
        density = round_to_1(1-gp/100)
        print("gp_" + str(gp) + "/" + file + ".smtx")

        if lib=="CLASP":
            cmd = "../build/src/benchmark_spmm --sparsity-type cvs --spmm CLASP --gemm cuBlas --precision "+ precision + " --block-size " + v + " --m " +  m + " --k " + k + " --n " + n + " --d " + str(density) + " --check "
        elif lib=="Sputnik":
            cmd = "../build/src/benchmark_spmm --sparsity-type csr --spmm sputnik --gemm cuBlas --precision "+ precision + " --m " +  m + " --k " + k + " --n " + n + " --d " + str(density) + " --check"

        os.system(cmd)


precision="half"
name=sys.argv[3]
bench_=sys.argv[4]
print(name)
print("__________")

if bench_=="rn50":
    for i in range(5):
        for d in dimensions:
            for bs in ["64", "128", "256"]:
                bench(name, name+"_i"+str(i), str(d[0]), str(d[1]), bs, sys.argv[1], precision, sys.argv[2])
elif bench_=="transformer":
    for i in range(5):
        for d in dim3:
            for bs in ["64", "128", "256"]:
                bench(name, name+"_i"+str(i), str(d[0]), str(d[1]), bs, sys.argv[1], precision, sys.argv[2])