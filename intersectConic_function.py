"""
-*- coding:utf-8 -*-
 @time :2019.05.22
 @IDE : pycharm
 @autor :juzh
 @email : jzh19980807@163.com
 """
import numpy as np
import math
from numpy.linalg import inv,det,matrix_rank


def completeIntersection(A,B):
    ee = np.dot(A,inv(-1*B))
    k=np.array([-1,
               np.trace(ee),
                -1*(det(ee[0:2,0:2])+det(ee[1:,1:])+det((ee[[0,2],:])[:,[0,2]])),
               det(ee)])
    r=np.roots(k)


    m=[]
    if np.isreal(r[0]):
        E0=A+r[0]*B
        E0=E0.astype(np.float32)
        m,l=decomposeDegenerateConic(E0)
    if len(m)==0 and np.isreal(r[1]):
        E0=A+r[1]*B
        E0=E0.astype(np.float32)
        m,l=decomposeDegenerateConic(E0)
    if len(m)==0 and np.isreal(r[2]):
        E0=A+r[2]*B
        E0=E0.astype(np.float32)
        m,l=decomposeDegenerateConic(E0)
        m=m.astype(np.float32)
        l=l.astype(np.float32)

    if len(m)==0:
        P=[]

        flag=0
    else:
        flag=1
        P1=intersectConicLine(A,m)
        P2=intersectConicLine(A,l)
        P=[]
        if P1 != []:
            P=P1[0]
            for i in range(len(P1)):
                if i==0:
                    continue
                else:
                    P=np.append(P,P1[i],axis=1)
        if P2 !=[]:
            if P ==[]:
                P=P2[0]
                for i in range(len(P2)):
                    if i ==0:
                        continue
                    else:
                        P=np.append(P,P2[i],axis=1)
            else:
                for i in range(len(P2)):
                    P = np.append(P, P2[i], axis=1)
        if P==[]:
            flag=0
    return P,flag

def max_with_index(L):
    maxV=0
    for i in L:
        if i>maxV:
            maxV=i

    di=[]
    for i in range(len(L)):
        if L[i]==maxV:
            di.append(i)
    return maxV,di


def decomposeDegenerateConic(c):
    if matrix_rank(c)==1:
        C=c
    else:
        B=-1*adjointSym3(c)
        temp_max=abs(np.diag(B))
        maxV,di=max_with_index(temp_max)#写的不符合规矩，只能先这样了
        i=di[0]
        if B[i,i]<0:
            l=[]
            m=[]
            return l,m
        else:
            b=math.sqrt(B[i,i])
            p=B[:,i]/b
            Mp=crossMatrix(p)
            C=c+Mp

    C_1=np.append(C[:,0],C[:,1],axis=0)
    C_1 = np.append(C_1, C[:, 2], axis=0)
    maxV,ci=max_with_index(abs(C_1))#写的不符合规矩，只能先这样了

    j=math.floor(ci[0]/3+1)
    i=ci[0]+1-(j-1)*3

    l=C[i-1,:].T
    m=C[:,j-1]

    return l,m


def intersectConicLine(A,l):
    P=[]
    p1,p2=getPointsOnline(l)

    p1Cp1=np.dot(np.dot(p1.T,A),p1)
    p2Cp2 = np.dot(np.dot(p2.T,A), p2)
    p1Cp2 = np.dot(np.dot(p1.T, A),p2)

    if p2Cp2==0:
        k1=-1*0.5*p1Cp1/p1Cp2
        P=p1+k1*p2
    else:
        delta=p1Cp2**2-p1Cp1*p2Cp2
        delta=delta.real
        if delta>=0:
            deltaSqrt=math.sqrt(delta)
            k1=(-1*p1Cp2+deltaSqrt)/p2Cp2
            k2=(-1*p1Cp2-deltaSqrt)/p2Cp2
            P=[p1+k1*p2,p1+k2*p2]
    return P


def crossMatrix(p):
    Mp=np.zeros((3,3))
    Mp[0,1]=p[2]
    Mp[0,2]=-1*p[1]
    Mp[1,0]=-1*p[2]
    Mp[1,2]=p[0]
    Mp[2,0]=p[1]
    Mp[2,1]=-1*p[0]
    return Mp


def adjointSym3(M):
    A=np.zeros((3,3))
    a,b,d,c,e,f=M[0,0],M[0,1],M[0,2],M[1,1],M[1,2],M[2,2]
    A[0,0]=c*f-e*e
    A[0,1]=-1*b*f+e*d
    A[0,2]=b*e-c*d

    A[1,0]=A[0,1]
    A[1,1]=a*f-d*d
    A[1,2]=-1*a*e+b*d

    A[2,0]=A[0,2]
    A[2,1]=A[1,2]
    A[2,2]=a*c-b*b
    return A


def getPointsOnline(L):
    if L[0]==0 and L[1]==0:
        p1=np.array([[1],[0],[0]])
        p2=np.array([[0],[1],[0]])
    else:
        p2=np.array([[-1*L[1]],[L[0]],[0]])
        if abs(L[0])<abs(L[1]):
            p1=np.array([[0],[-1*L[2]],[L[1]]])
        else:
            p1=np.array([[-1*L[2]],[0],[L[0]]])
    return p1,p2


def intersectConic(A,B):
    r1=matrix_rank(A)
    r2=matrix_rank(B)

    if r1==3 and r2==3:
        P,flag=completeIntersection(A,B)
    else:
        if r2<3:
            defE=B
            fullE=A
        else:
            defE=A
            fullE=B
        flag=1
        m,l=decomposeDegenerateConic(defE)
        P1=intersectConicLine(fullE,m)
        P2=intersectConicLine(fullE,l)
        P=[]
        if P1 != []:
            P=P1[0]
            for i in range(len(P1)):
                if i==0:
                    continue
                else:
                    P=np.append(P,P1[i],axis=1)
        if P2 !=[]:
            if P ==[]:
                P=P2[0]
                for i in range(len(P2)):
                    if i ==0:
                        continue
                    else:
                        P=np.append(P,P2[i],axis=1)
            else:
                for i in range(len(P2)):
                    P = np.append(P, P2[i], axis=1)


    return P,flag


