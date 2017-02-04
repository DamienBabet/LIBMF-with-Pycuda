# -*- coding: utf-8 -*-
#import pyquickhelper, pyensae
import random
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from scipy import sparse as sp

p = np.array([1,2,3,4,5,0,0,2])
p = p.astype(np.float32)
q = np.array([8,7,6,5,0,3,2,1])
q = q.astype(np.float32)
r = np.array([0,0,12,0,1,7,2,2,0.1,3,3,3.4])
r = r.astype(np.float32)
v = np.array([0,2,4])
v = v.astype(np.float32)
Knum = 2
tP = 2
tQ = 2
gamma = 0.1
lambdaP = 0.1
lambdaQ = 0.1

kernel_code_template = """
    __global__ void block_update(float *p, float *q, float *r, int *v)
    {
      const unsigned int K = %(Knum)s;
      const float gamma = %(gamma)s;
      const float lambdaP = %(lambdaP)s;
      const float lambdaQ = %(lambdaQ)s;  
      int idx = threadIdx.x;
      int rstart = v[idx];
      int rend = v[idx+1];
      float pqij = 0;
      float p_temp = 0;
      for (int n = 0; n < rend-rstart; ++n){
        int i = r[3 * (n+rstart)];
        int j = r[(3 * (n+rstart)) + 1];
        float rij = r[(3 * (n+rstart)) + 2];
        pqij = 0;
        for (int k = 0; k < K; ++k){
            pqij += p[(i*K)+k] * q[(j*K)+k];}
        float eij = rij - pqij;
        for (int k = 0; k < K; ++k){
            p_temp = gamma * eij * q[(j*K)+k] - gamma * lambdaP * p[(i*K)+k];
            q[(j*K)+k] += gamma * eij * p[(i*K)+k] - gamma * lambdaQ * q[(j*K)+k];
            p[(i*K)+k] += p_temp;}
      }
    }
    """
    
kernel_code = kernel_code_template % { 
    'Knum': Knum,
    'gamma': gamma,
    'lambdaP': lambdaP,
    'lambdaQ': lambdaQ,
    }

mod = compiler.SourceModule(kernel_code)


func = mod.get_function("block_update")

p_gpu = cuda.mem_alloc(p.nbytes)
cuda.memcpy_htod(p_gpu, p)
q_gpu = cuda.mem_alloc(q.nbytes)
cuda.memcpy_htod(q_gpu, q)
r_gpu = cuda.mem_alloc(r.nbytes)
cuda.memcpy_htod(r_gpu, r)
v_gpu = cuda.mem_alloc(v.nbytes)
cuda.memcpy_htod(v_gpu, v)



func(p_gpu, q_gpu, r_gpu, v_gpu, block=(4,4,1))

p_changed = np.empty_like(p)
cuda.memcpy_dtoh(p_changed, p_gpu)
q_changed = np.empty_like(q)
cuda.memcpy_dtoh(q_changed, q_gpu)
r_changed = np.empty_like(r)
cuda.memcpy_dtoh(r_changed, r_gpu)

print p
print p_changed

print q
print q_changed

print r
print r_changed