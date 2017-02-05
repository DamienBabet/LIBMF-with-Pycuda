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

nbUsers=9
nbFilms=9
nbBloc=3
Knum=1
nbIter=3
gamma=0.001
lambdaP=0.1
lambdaQ=0.1

AjoutUsersFictifs=int(np.ceil(np.true_divide(nbUsers,nbBloc))*nbBloc-nbUsers)
AjoutFilmsFictifs=int(np.ceil(np.true_divide(nbFilms,nbBloc))*nbBloc-nbFilms)

p=np.random.uniform(9,10,(((nbUsers+AjoutUsersFictifs)*Knum)))
q=np.random.uniform(9,10,(((nbFilms+AjoutFilmsFictifs)*Knum)))

tp=int((nbUsers+AjoutUsersFictifs)/nbBloc)
tq=int((nbFilms+AjoutFilmsFictifs)/nbBloc)

r_sparse = sp.rand(nbUsers+AjoutUsersFictifs,nbFilms+AjoutFilmsFictifs,0.6)
r_data=np.transpose(np.array([r_sparse.row, r_sparse.col, r_sparse.data]))
df=pd.DataFrame(r_data, columns=("i","j","value"))
df["numBlocLigne"] =df.apply(lambda row : int(row[0]/tp), axis = 1 )
df["numBlocColonne"] =df.apply(lambda row : int(row[1]/tq), axis = 1 )
v = pd.DataFrame(df.groupby(["numBlocLigne", "numBlocColonne"]).count())
v = v.reset_index()[["numBlocLigne","numBlocColonne","value"]]

def erreur(x,y,z):
    error = 0
    for i in (0,len(z)-1):
        eij = z[i,2]-x[z[i,0]]*y[z[i,1]]
        error += eij*eij
    print(error)

erreur(p,q,r_data)

print(p)
print(q)
print(df)
print(v)

mod = SourceModule("""
    __global__ void block_update(float *p, float *q, float *r, float *v)
    {
      int K = 1;
      float gamma = 0.5;
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

func = mod.get_function("block_update")


for l in range(1,nbIter):
    #Choisir une permutation des colonnes
    L=np.array(range(nbBloc))
    random.shuffle(L)
    
    # Mise en forme de R selon l'ordre de la permutation et création de v_permut (vecteur des coordonnees des debuts de blocks sur R)
    r_permut=np.array(df[(df["numBlocLigne"]==0) & (df["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    v_permut=np.array(v[(v["numBlocLigne"]==0) & (v["numBlocColonne"]==L[0])][[2]]).flatten()
    for i in range(1,nbBloc) :
        r_permut=np.append(r_permut,np.array(df[(df["numBlocLigne"]==i) & (df["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
        v_permut=np.append(v_permut,np.array(v[(v["numBlocLigne"]==i) & (v["numBlocColonne"]==L[i])][[2]]).flatten()).flatten()
    v_permut_cum=np.append(0,np.cumsum(v_permut))
    
    # Execution du kernel
    func(cuda.InOut(p), cuda.InOut(q), cuda.InOut(r_permut), cuda.InOut(v_permut_cum), block=(4,4,1))
    print(p)
    
print(p)
print(q)

erreur(p,q,r_data)
