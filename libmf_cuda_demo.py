# -*- coding: utf-8 -*-
# Prototype demontrant la convergence de l'algorithme sur un cas, via CUDA
 
import random
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda import compiler, gpuarray, tools
from pycuda.compiler import SourceModule
from scipy import sparse as sp

# On definit les deux parametres a retrouver
p0=np.array([1,2,4,1,6,1,1,3,2])
q0=np.array([1,1,1,2,5,1,2,1,1])

p2=p0.reshape((9,1))
q2=q0.reshape((1,9))
r=np.dot(p2,q2)
# La matrice R est le produit des vrais P et Q
r


# On initialise l'algo avec des P et Q aleatoires et eloignes des vraies valeurs
p=np.random.uniform(9,10,9)
p = p.astype(np.float32)
q=np.random.uniform(9,10,9)
q = q.astype(np.float32)

# Les array des blocs pour l'algo LIBMF ont ete construit a la main...
r1 = np.array([0,0,1,0,1,1,2,2,4,3,3,2,4,3,12,4,4,30,5,5,1,6,7,1,6,8,1,8,6,4])
r1 = r1.astype(np.float32)
v1 = np.array([0,3,7,10])
v1 = v1.astype(np.float32)

r2 = np.array([0,3,2,1,3,4,1,4,10,2,4,20,2,5,4,3,6,2,6,0,1,7,0,3,8,0,2])
r2 = r2.astype(np.float32)
v2 = np.array([0,5,6,9])
v2 = v2.astype(np.float32)

r3 = np.array([0,6,2,1,8,2,1,7,2,3,0,1,4,1,6,6,4,5])
r3 = r3.astype(np.float32)
v3 = np.array([0,3,5,6])
v3 = v3.astype(np.float32)

# Parametrisation de l'algo
Knum = 1
gamma = 0.01
lambdaP = 0.1
lambdaQ = 0.1
nbIter = 6000
p_start = p
q_start = q

print p_start
print q_start

# On mesure l'erreur quadratique totale avant et apres l'algo
def erreur(x,y,z):
    error = 0
    for i in (0,len(z)-1):
        eij = z[i,2]-x[z[i,0]]*y[z[i,1]]
        error += eij*eij
    print(error)

erreur(p,q,r)

# Le kernel Cuda sous C :
mod = SourceModule("""
    __global__ void block_update(float *p, float *q, float *r, float *v)
    {
      int K = 1;
      float gamma = 0.001;
      float lambdaP = 0.1;
      float lambdaQ = 0.1;
      int idx = threadIdx.x;
      int rstart = v[idx];
      int rend = v[idx+1];
      float pqij = 0;
      float p_temp = 0;
      float q_temp = 0;
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
            q_temp = gamma * eij * p[(i*K)+k] - gamma * lambdaQ * q[(j*K)+k];
            if ((q[(j*K)+k] + q_temp) > 0)
                q[(j*K)+k] += q_temp;
            if ((p[(i*K)+k] + p_temp) > 0)
                p[(i*K)+k] += p_temp;}
      }
    }
    """)

# On definit la fonction python qui fera appel au kernel
func = mod.get_function("block_update")

# La boucle principale de l'algorithme, qui tourne sur les 
# blocs predefinis "a la main"
for l in range(nbIter):    
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r1), cuda.InOut(v1), block=(3,3,1))
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r2), cuda.InOut(v2), block=(3,3,1))
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r3), cuda.InOut(v3), block=(3,3,1))

# On mesure a nouveau l'erreur : elle a beaucoup diminue
erreur(p,q,r)

# On peut comparer le P estime au vrai parametre pour constater la convergence
print(p)
print(p0)