import numpy as np
from numpy.linalg import inv,det,eig,qr,svd
import math

def completeIntersection(E1,E2):
    E2_matrix=np.mat(E2)
    E1_matrix=E1
    EE=E1_matrix*np.linalg.inv(-1*E2_matrix)
    EE_13=np.mat([[EE[0,0],EE[0,2]],[EE[2,2],EE[2,2]]])
    k=np.array([-1,
                np.trace(EE),
               -1*(det(EE[0:2,0:2])+det(EE[1:,1:])+det(EE_13)),
                det(EE)])
    r=np.root(k)
    m=[]
    if isinstance(r[0],(int,float)):
        E0=E1+r[0]*E2
        m,l=decomposeDegenerateConic(E0)
    if (isempty(m) and isinstance(r[1],(int,float))):
        E0 = E1 + r[1] * E2
        m, l = decomposeDegenerateConic(E0)
    if (isempty(m) and isinstance(r[2], (int, float))):
        E0 = E1 + r[2] * E2
        m, l = decomposeDegenerateConic(E0)

    if isempty(m):
        p=[]
        return P
    P1=intersection(E1,m)
    P2=intersection(E1,l)
    P=[P1,P2]
    return P

def decomposeDegenerateConic(E0):
    if np.linalg.matrix_rank(E0)==1:
        C=E0#如果秩是1，可以直接分成两条直线
    else:
        B=-1*adjontSym3(E0)
        maxV,di=max(abs(numpy.diag(B)))
        i=di(1)#不知道是怎么写的，明天用matlab跑一下看看di是什么意思
        if B[i,i]<0:
            l,m=[],[]
            return l,m
        else:
            b=pow(B[i,i],0.5)
            p=B[:,i]/b

            Mp=crossMatrix(p)
            C=E0+Mp
    maxV,ci=max(np.abs(C))#注意和MATLAB对比
    j=math.floor((ci(1)-1)/3)+1
    i=ci(1)-(j-1)*3
    l=np.transpose(C[i,:])
    m=C[:,j]
    return l,m