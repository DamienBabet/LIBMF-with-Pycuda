# -*- coding: utf-8 -*-
#import pyquickhelper, pyensae
import random
import numpy as np
import pandas as pd
from scipy import sparse as sp

nbUsers=9
nbFilms=9
nbBloc=3
Knum=1
nbIter=200
gamma=0.05
lambdaP=0.1
lambdaQ=0.1

AjoutUsersFictifs=int(np.ceil(np.true_divide(nbUsers,nbBloc))*nbBloc-nbUsers)
AjoutFilmsFictifs=int(np.ceil(np.true_divide(nbFilms,nbBloc))*nbBloc-nbFilms)

p=np.random.uniform(9,10,(((nbUsers+AjoutUsersFictifs)*Knum)))
q=np.random.uniform(0,1,(((nbFilms+AjoutFilmsFictifs)*Knum)))

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
    
p_start = p
q_start = q
print(p==p_start)
print(q==q_start)
print(q)
print(p)
print(df)
print(v)

for l in range(nbIter):
    
    #Choisir une permutation des colonnes
    L=np.array(range(nbBloc))
    random.shuffle(L)
    
    # Mise en forme de R selon l'ordre de la permutation et création de v_permut (vecteur des coordonnees des debuts de blocks sur R)
    r_permut=np.array(df[(df["numBlocLigne"]==0) & (df["numBlocColonne"]==L[0])][[0,1,2]]).flatten()
    v_permut=np.array(v[(v["numBlocLigne"]==0) & (v["numBlocColonne"]==L[0])][[2]]).flatten()
    for i in range(1,nbBloc) :
        r_permut=np.append(r_permut,np.array(df[(df["numBlocLigne"]==i) & (df["numBlocColonne"]==L[i])][[0,1,2]]).flatten()).flatten()
        v_permut=np.append(v_permut,np.array(v[(v["numBlocLigne"]==i) & (v["numBlocColonne"]==L[i])][[2]]).flatten()).flatten()
    v_permut_cum=np.append(1,np.cumsum(v_permut))
    
    # Execution du kernel
    K = Knum
    for t in range(1,nbBloc):
        idx = t
        rstart = v_permut_cum[idx]-1
        rend = v_permut_cum[idx+1]-1
        p_temp = 0
        q_temp = 0
        for n in (0,rend-rstart):
            i = r_permut[3 * (n+rstart)]
            j = r_permut[(3 * (n+rstart)) + 1]
            rij = r_permut[(3 * (n+rstart)) + 2]
            pqij = 0
            for k in (0,K-1):
                pqij += p[(i*K)+k] * q[(j*K)+k]
            eij = rij - pqij
            for k in (0,K-1):
                p_temp = gamma * eij * q[(j*K)+k] - gamma * lambdaP * p[(i*K)+k]
                q_temp = gamma * eij * p[(i*K)+k] - gamma * lambdaQ * q[(j*K)+k]
                if (q[(j*K)+k] + q_temp) > 0:
                    q[(j*K)+k] += q_temp
                if (p[(i*K)+k] + p_temp) > 0:
                    p[(i*K)+k] += p_temp
    
    
print(q)
print(p)
print(p==p_start)
print(q==q_start)

erreur(p,q,r_data)