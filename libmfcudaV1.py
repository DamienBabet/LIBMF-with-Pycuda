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

nbUsers=22
nbFilms=8
nbBloc=2
Knum=2
nbIter=10
gamma=0.001
lambdaP=0.1
lambdaQ=0.1

AjoutUsersFictifs=int(np.ceil(np.true_divide(nbUsers,nbBloc))*nbBloc-nbUsers)
AjoutFilmsFictifs=int(np.ceil(np.true_divide(nbFilms,nbBloc))*nbBloc-nbFilms)

p=np.random.uniform(0,1,(((nbUsers+AjoutUsersFictifs)*Knum)))
q=np.random.uniform(0,1,(((nbFilms+AjoutFilmsFictifs)*Knum)))

p2=p.reshape((nbUsers+AjoutUsersFictifs,Knum))
q2=q.reshape((Knum,nbFilms+AjoutFilmsFictifs))
r=np.dot(p2,q2).T

tp=int((nbUsers+AjoutUsersFictifs)/nbBloc)
tq=int((nbFilms+AjoutFilmsFictifs)/nbBloc)

r_sparse = sp.rand(nbUsers+AjoutUsersFictifs,nbFilms+AjoutFilmsFictifs,0.6)
r_data=np.transpose(np.array([r_sparse.row, r_sparse.col, r_sparse.data]))
df=pd.DataFrame(r_data, columns=("i","j","value"))
df["numBlocLigne"] =df.apply(lambda row : int(row[0]/tp), axis = 1 )
df["numBlocColonne"] =df.apply(lambda row : int(row[1]/tq), axis = 1 )
v = pd.DataFrame(df.groupby(["numBlocLigne", "numBlocColonne"]).count())
v = v.reset_index()[["numBlocLigne","numBlocColonne","value"]]

print(q)

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
    v_permut=np.array(v[(v["numBlocLigne"]==0) & (v["numBlocColonne"]==L[0])][[2]]).flatten()
    for i in range(1,nbBloc) :
        r_permut=np.append(r_permut,np.array(df[(df["numBlocLigne"]==i) & (df["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
        v_permut=np.append(v_permut,np.array(v[(v["numBlocLigne"]==i) & (v["numBlocColonne"]==L[i])][[2]]).flatten()).flatten()
    v_permut_cum=np.append(0,np.cumsum(v_permut))
    
      
    # Transfert des données sur la GPU
    #p_gpu = cuda.mem_alloc(p.nbytes)
    #cuda.memcpy_htod(p_gpu, p)
    #q_gpu = cuda.mem_alloc(q_permut.nbytes)
    #cuda.memcpy_htod(q_gpu, q_permut)
    #r_gpu = cuda.mem_alloc(r_permut.nbytes)
    #cuda.memcpy_htod(r_gpu, r_permut)
    #v_gpu = cuda.mem_alloc(v_permut.nbytes)
    #cuda.memcpy_htod(v_gpu, v_permut)
    # Mettre __syncthreads();  dans le code du kernel
    
    # Execution du kernel
    #func(cuda.InOut(p), cuda.InOut(q_permut), cuda.InOut(r_permut), cuda.InOut(v_permut), block=(nbBloc,1,1))
    
    # Recuperation des resultats
    #p_changed = np.empty_like(p)
    #cuda.memcpy_dtoh(p_changed, p_gpu)      

    #q_permut = q_changed
    #p = p_changed
    
    #Remettre Q dans l'ordre
    q2=q_permut.reshape((Knum,nbFilms+AjoutFilmsFictifs))
    qEnBloc=q2.T.reshape(nbBloc,int((nbFilms+AjoutFilmsFictifs)/nbBloc),Knum)
    q=qEnBloc[int(np.floor(np.where(L==0)))].tolist()
    for i in range(1,nbBloc) :
        q=np.append(q,qEnBloc[int(np.floor(np.where(L==i)))].tolist())
    print(q_permut)
    
    
#q_changed = np.empty_like(q_permut)
#cuda.memcpy_dtoh(q_changed, q_gpu)      

