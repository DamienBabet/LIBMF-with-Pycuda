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

p0=np.array([1,0,0,1,1,1,1,3])
q0=np.array([1,1,1,2,0,1,2,1])

p2=p0.reshape((4,2))
q2=q0.reshape((2,4))
r=np.dot(p2,q2)
r

p = np.array([1,2,3,4,5,0,0,2])
p = p.astype(np.float32)
q = np.array([8,7,1,5,0,3,2,1])
q = q.astype(np.float32)
r1 = np.array([0,0,1,0,1,1,2,2,3,3,3,5])
r1 = r1.astype(np.float32)
v1 = np.array([0,2,4])
v1 = v1.astype(np.float32)

r2 = np.array([0,2,1,1,2,2,1,3,1,2,0,1,2,1,2,3,1,4])
r2 = r2.astype(np.float32)
v2 = np.array([0,3,6])
v2 = v2.astype(np.float32)

Knum = 2
tP = 2
tQ = 2
gamma = 0.01
lambdaP = 0.1
lambdaQ = 0.1
nbIter = 30
p_start = p
q_start = q

print p_start
print q_start


mod = SourceModule("""
    __global__ void block_update(float *p, float *q, float *r, float *v)
    {
      int K = 2;
      float gamma = 0.001;
      float lambdaP = 0.1;
      float lambdaQ = 0.1;
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
            p_temp =  gamma * eij * q[(j*K)+k] - gamma * lambdaP * p[(i*K)+k];
            q[(j*K)+k] += gamma * eij * p[(i*K)+k] - gamma * lambdaQ * q[(j*K)+k];
            p[(i*K)+k] += p_temp;}
      }
    }
    """)

func = mod.get_function("block_update")

#p_gpu = cuda.mem_alloc(p.nbytes)
#cuda.memcpy_htod(p_gpu, p)
#q_gpu = cuda.mem_alloc(q.nbytes)
#cuda.memcpy_htod(q_gpu, q)
#r_gpu = cuda.mem_alloc(r.nbytes)
#cuda.memcpy_htod(r_gpu, r)
#v_gpu = cuda.mem_alloc(v.nbytes)
#cuda.memcpy_htod(v_gpu, v)


for l in range(nbIter):    
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r1), cuda.InOut(v1), block=(4,4,1))
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r2), cuda.InOut(v2), block=(4,4,1))

    print p
    print q
    print r1
    print v1
    print r2
    print v2

#q = np.array([q[4],q[5],q[6],q[7],q[0],q[1],q[2],q[3]])
#q = q.astype(np.float32)

#p_changed = np.empty_like(p)
#cuda.memcpy_dtoh(p_changed, p_gpu)
#q_changed = np.empty_like(q)
#cuda.memcpy_dtoh(q_changed, q_gpu)
#r_changed = np.empty_like(r)
#cuda.memcpy_dtoh(r_changed, r_gpu)