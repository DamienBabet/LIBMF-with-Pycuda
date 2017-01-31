import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import numpy
n = 100
K=3
Petoile = numpy.random.uniform(0,1,n,K)
Qetoile = numpy.random.uniform(0,1,K,n)

r = Petoile * Qetoile

a = a.astype(numpy.float32)

a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)

cuda.memcpy_htod(a_gpu, a)

mod = SourceModule("""
    __global__ void doublify(float *p, float *q, float *r, int t, int K, int gamma, int lambdaP, int lambdaQ)
    {
      int idx = threadIdx.x;
      int start = idx * t * K;
      int end = (idx + 1) * t * K - 1;
      int rstart = idx * (t * t);
      int rend = (idx + 1) * t * t - 1;
      int 
      for (int u = 0, u < t, ++u)
        for (int v = 0, v < t, ++v){
            for (int k = 0; k < K; ++k)
                pquv += p[start+(u*K)+k] * q[start+(v*K)+k];
            euv = r[rstart+t*u+v] - pquv;
            for (int k = 0; k < K; ++k)
                p[u][k] += gamma * e[u][v] * q[k][v] - gamma * lambdaP * p[u][k];
            for (int k = 0; k < K; ++k)
                q[k][v] += gamma * e[u][v] * p[u][k] - gamma * lambdaQ * q[k][v];
        }
    }
    """)

func = mod.get_function("doublify")
func(a_gpu, block=(4,4,1))

a_doubled = numpy.empty_like(a)
cuda.memcpy_dtoh(a_doubled, a_gpu)
print "original array:"
print a
print "doubled with kernel:"
print a_doubled

# alternate kernel invocation -------------------------------------------------

func(cuda.InOut(a), block=(4, 4, 1))
print "doubled with InOut:"
print a

# part 2 ----------------------------------------------------------------------

import pycuda.gpuarray as gpuarray
a_gpu = gpuarray.to_gpu(numpy.random.randn(4,4).astype(numpy.float32))
a_doubled = (2*a_gpu).get()

print "original array:"
print a_gpu
print "doubled with gpuarray:"
print a_doubled
