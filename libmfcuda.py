import pyquickhelper, pyensae
import random
import numpy as np
import pandas as pd
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from scipy import sparse as sp

nbUsers=100
nbFilms=55
nbBloc=10
K=3
nbIter=2
gamma=0.002
lambdaP=0.1
lambdaQ=0.1

AjoutUsersFictifs=int(nbUsers-np.floor(nbUsers/nbBloc)*nbBloc)
AjoutFilmsFictifs=int(nbFilms-np.floor(nbFilms/nbBloc)*nbBloc)

p=np.random.uniform(0,1,(((nbUsers+AjoutUsersFictifs)*K)))
q=np.random.uniform(0,1,(((nbFilms+AjoutFilmsFictifs)*K)))

p2=p.reshape((nbUsers+AjoutUsersFictifs,K))
q2=q.reshape((K,nbFilms+AjoutFilmsFictifs))
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
    __global__ void block_update(float *p, float *q, float *r, float *v, int tP, int tQ, int K, int gamma, int lambdaP, int lambdaQ)
    {
      int idx = threadIdx.x;
      int startP = idx * tP * K;
      int endP = (idx + 1) * tP * K - 1;
      int startQ = idx * tQ * K;
      int endQ = (idx + 1) * tQ * K - 1;
      int rstart = v[idx];
      int rend = v[idx+1];
      float p_temp[K];
      float q_temp[K];
      float pqij;
      for (int r = rstart, r < rend, ++r){
        int i = v[3 * r];
        int j = v[(3 * r) + 1];
        float rij = v[(3 * r) + 2];
        pqij = 0;
        for (int k = 0; k < K; ++k){
          p_temp[k] = p[startP+(i*K)+k];
          q_temp[k] = q[startQ+(j*K)+k];
          pqij += p_temp[k] * q_temp[k];}
        float eij = rij - pqij;
        for (int k = 0; k < K; ++k){
          p[start+(i*K)+k] += gamma * eij * q_temp[k] - gamma * lambdaP * p_temp[k];
          q[start+(j*K)+k] += gamma * eij * p_temp[k] - gamma * lambdaQ * q_temp[k];}
      }
    }
    """)

func = mod.get_function("block_update")

for l in range(nbIter):
    
    #Choisir une permutation des colonnes
    L=np.array(range(nbBloc))
    random.shuffle(L)
    
    # Mise en forme de Q selon l'ordre de la permutation
    q2=q.reshape((K,nbFilms+AjoutFilmsFictifs))
    qEnBloc=q2.T.reshape(nbBloc,int((nbFilms+AjoutFilmsFictifs)/nbBloc),K)
    q_permut=qEnBloc[L[0]].tolist()
    for i in range(1,nbBloc) :
        q_permut=np.append(q_permut,qEnBloc[L[i]].tolist())
    
    # Mise en forme de R selon l'ordre de la permutation et création de v_permut (vecteur des coordonnees des debuts de blocks sur R)
    r_permut=np.array(df[(df["numBlocLigne"]==0) & (df["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    v_permut=np.array(v[(v["numBlocLigne"]==0) & (v["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    for i in range(1,nbBloc) :
        r_permut=np.append(r_permut,np.array(df[(df["numBlocLigne"]==i) & (df["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
        v_permut=np.append(v_permut,np.array(v[(v["numBlocLigne"]==i) & (v["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
      
    # Transfert des données sur la GPU
    p_gpu = cuda.mem_alloc(p.nbytes)
    cuda.memcpy_htod(p_gpu, p)
    q_gpu = cuda.mem_alloc(q_permut.nbytes)
    cuda.memcpy_htod(q_gpu, q_permut)
    r_gpu = cuda.mem_alloc(r_permut.nbytes)
    cuda.memcpy_htod(r_gpu, r_permut)
    
    # Mettre __syncthreads();  dans le code du kernel
    
    # Execution du kernel
    func(p_gpu, q_gpu, r_gpu, v_gpu, tP, tQ, K, gamma, lambdaP, lambdaQ)

    #Remettre Q dans l'ordre
    q2=q_permut.reshape((K,nbFilms+AjoutFilmsFictifs))
    qEnBloc=q2.T.reshape(nbBloc,int((nbFilms+AjoutFilmsFictifs)/nbBloc),K)
    q=qEnBloc[int(np.floor(np.where(L==0)))].tolist()
    for i in range(1,nbBloc) :
        q=np.append(q,qEnBloc[int(np.floor(np.where(L==i)))].tolist())
    
    
