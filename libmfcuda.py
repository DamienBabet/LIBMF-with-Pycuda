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

nbUsers=100
nbFilms=56
nbBloc=10
Knum=3
nbIter=2
gamma=0.002
lambdaP=0.1
lambdaQ=0.1

AjoutUsersFictifs=int(np.ceil(np.true_divide(nbUsers,nbBloc))*nbBloc-nbUsers)
AjoutFilmsFictifs=int(np.ceil(np.true_divide(nbFilms,nbBloc))*nbBloc-nbFilms)

p=np.random.uniform(0,1,(((nbUsers+AjoutUsersFictifs)*Knum)))
q=np.random.uniform(0,1,(((nbFilms+AjoutFilmsFictifs)*Knum)))

tP=(nbUsers+AjoutUsersFictifs)/nbBloc
tQ=(nbFilms+AjoutFilmsFictifs)/nbBloc

p2=p.reshape((nbUsers+AjoutUsersFictifs,Knum))
q2=q.reshape((Knum,nbFilms+AjoutFilmsFictifs))
r=np.dot(p2,q2).T

tp=int((nbUsers+AjoutUsersFictifs)/nbBloc)
tq=int((nbFilms+AjoutFilmsFictifs)/nbBloc)

r_sparse = sp.rand(nbUsers+AjoutUsersFictifs,nbFilms+AjoutFilmsFictifs,0.1)
r_data=np.transpose(np.array([r_sparse.row, r_sparse.col, r_sparse.data]))
df=pd.DataFrame(r_data, columns=("i","j","value"))
df["numBlocLigne"] =df.apply(lambda row : int(row[0]/tp), axis = 1 )
df["numBlocColonne"] =df.apply(lambda row : int(row[1]/tq), axis = 1 )
v = pd.DataFrame(df.groupby(["numBlocLigne", "numBlocColonne"]).count())
v = v.reset_index()[["numBlocLigne","numBlocColonne","value"]]
v

mod = SourceModule("""
    __global__ void block_update(float *p, float *q, float *r, int *v, int K, int tP, int tQ, int gamma, int lambdaP, int lambdaQ)
    {
      int idx = threadIdx.x;
      int startP = idx * tP * K;
      int endP = (idx + 1) * tP * K - 1;
      int startQ = idx * tQ * K;
      int endQ = (idx + 1) * tQ * K - 1;
      int rstart = v[idx];
      int rend = v[idx+1];
      float pqij = 0;
      float p_temp = 0;
      for (int n = 0; n < rend-rstart; ++n){
        int i = v[3 * (n+rstart)];
        int j = v[(3 * (n+rstart)) + 1];
        float rij = v[(3 * (n+rstart)) + 2];
        pqij = 0;
        if (i < endP && j < endQ) {
            for (int k = 0; k < K; ++k){
                pqij += p[startP+(i*K)+k] * q[startQ+(j*K)+k];}
            float eij = rij - pqij;
            for (int k = 0; k < K; ++k){
                p_temp = gamma * eij * q[startQ+(j*K)+k] - gamma * lambdaP * p[startP+(i*K)+k];
                q[startQ+(j*K)+k] += gamma * eij * p[startP+(i*K)+k] - gamma * lambdaQ * q[startQ+(j*K)+k];
                p[startP+(i*K)+k] += p_temp;}
        }      
      }
    }
    """)

func = mod.get_function("block_update")

for l in range(nbIter):
    
    #Choisir une permutation des colonnes
    L=np.array(range(nbBloc))
    random.shuffle(L)
    
    # Mise en forme de Q selon l'ordre de la permutation
    q2=q.reshape((Knum,nbFilms+AjoutFilmsFictifs))
    qEnBloc=q2.T.reshape(nbBloc,int((nbFilms+AjoutFilmsFictifs)/nbBloc),Knum)
    q_permut=qEnBloc[L[0]].tolist()
    for i in range(1,nbBloc) :
        q_permut=np.append(q_permut,qEnBloc[L[i]].tolist())
    
    # Mise en forme de R selon l'ordre de la permutation et création de v_permut (vecteur des coordonnees des debuts de blocks sur R)
    r_permut=np.array(df[(df["numBlocLigne"]==0) & (df["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    v_permut=np.array(v[(v["numBlocLigne"]==0) & (v["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    for i in range(1,nbBloc) :
        r_permut=np.append(r_permut,np.array(df[(df["numBlocLigne"]==i) & (df["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
        v_permut=np.append(v_permut,np.array(v[(v["numBlocLigne"]==i) & (v["numBlocColonne"]==L[i])][[2]]).flatten()).flatten()
    v_permut_cum=np.append(0,np.cumsum(v_permut.reshape(nbBloc,3).T[2][range(0,len(v_permut.reshape(nbBloc,3).T[2])-1)]))
    
    # Transfert des données sur la GPU
    p_gpu = cuda.mem_alloc(p.nbytes)
    cuda.memcpy_htod(p_gpu, p)
    q_gpu = cuda.mem_alloc(q_permut.nbytes)
    cuda.memcpy_htod(q_gpu, q_permut)
    r_gpu = cuda.mem_alloc(r_permut.nbytes)
    cuda.memcpy_htod(r_gpu, r_permut)
    v_gpu = cuda.mem_alloc(v_permut_cum.nbytes)
    cuda.memcpy_htod(v_gpu, v_permut_cum)
    # Mettre __syncthreads();  dans le code du kernel
    
    # Execution du kernel
    func(p_gpu, q_gpu, r_gpu, v_gpu, Knum, tP, tQ, gamma, lambdaP, lambdaQ, block=(nbBloc,1,1))

    #Remettre Q dans l'ordre
    q2=q_permut.reshape((Knum,nbFilms+AjoutFilmsFictifs))
    qEnBloc=q2.T.reshape(nbBloc,int((nbFilms+AjoutFilmsFictifs)/nbBloc),Knum)
    q=qEnBloc[int(np.floor(np.where(L==0)))].tolist()
    for i in range(1,nbBloc) :
        q=np.append(q,qEnBloc[int(np.floor(np.where(L==i)))].tolist())
