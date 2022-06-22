# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import collect_env as collect_base_env
from mmcv.utils import get_git_hash

import mmedit


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMEditing'] = f'{mmedit.__version__}+{get_git_hash()[:7]}'

    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print('{}: {}'.format(name, val))


# (mmedit) [root@localhost utils]# python collect_env.py 
# sys.platform: linux
# Python: 3.7.11 (default, Jul 27 2021, 14:32:16) [GCC 7.5.0]
# CUDA available: True
# GPU 0,1,2,3,4,5,6,7: NVIDIA GeForce RTX 2080
# CUDA_HOME: /usr/local/cuda
# NVCC: Build cuda_11.3.r11.3/compiler.29920130_0
# GCC: gcc (GCC) 5.3.0
# PyTorch: 1.10.2
# PyTorch compiling details: PyTorch built with:
  # - GCC 7.3
  # - C++ Version: 201402
  # - Intel(R) oneAPI Math Kernel Library Version 2021.4-Product Build 20210904 for Intel(R) 64 architecture applications
  # - Intel(R) MKL-DNN v2.2.3 (Git Hash 7336ca9f055cf1bfa13efb658fe15dc9b41f0740)
  # - OpenMP 201511 (a.k.a. OpenMP 4.5)
  # - LAPACK is enabled (usually provided by MKL)
  # - NNPACK is enabled
  # - CPU capability usage: AVX512
  # - CUDA Runtime 11.3
  # - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_37,code=compute_37
  # - CuDNN 8.2
  # - Magma 2.5.2
  # - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.3, CUDNN_VERSION=8.2.0, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DEDGE_PROFILER_USE_KINETO -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.10.2, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 

# TorchVision: 0.11.3
# OpenCV: 4.5.5
# MMCV: 1.3.9
# MMCV Compiler: GCC 5.3
# MMCV CUDA Compiler: 11.3
# MMEditing: 0.11.0+4075d92
