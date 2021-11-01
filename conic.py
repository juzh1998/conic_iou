import numpy as np
import cmath
import math
from scipy.linalg import sqrtm,fractional_matrix_power
from numpy.linalg import inv,det,eig,qr,svd

pi=3.14159265
'''
def para2mat(conic):
    #将参数转变为椭圆二次型
    c_x,c_y,a,b,th=conic[0],conic[1],conic[2],conic[3],conic[4]
    cos_val=math.cos(th)
    sin_val=math.sin(th)
    a_11=(b**2)*(cos_val**2)+(a**2)*(sin_val**2)
    a_22=(b**2)*(sin_val**2)+(a**2)*(cos_val**2)
    a_33=((a**2)+(b**2))*((c_x*cos_val-c_y*sin_val)**2)\
            -(a**2)*(b**2)
    a_12=cos_val*sin_val*((b**2)-(a**2))
    a_21=a_12
    a_13=(-1)*(c_x*(cos_val**2)*(b**2)+c_x*(sin_val**2)*(a**2)\
        +c_y*cos_val*sin_val*(b**2-a**2))
    a_31=a_13
    a_23=(-1)*(c_y*(cos_val**2)*(b**2)+c_y*(cos_val**2)*(a**2)+c_x*cos_val*sin_val*(b**2-a**2))
    a_32=a_23
    mat=np.array([[a_11,a_12,a_23],[a_21,a_22,a_23],[a_31,a_32,a_33]])
    return mat
'''
def para2mat(conic):
    c_x,c_y,a,b,th=conic[0],conic[1],conic[2],conic[3],conic[4]
    cos_val,sin_val=math.cos(th),math.sin(th)
    transform_mat=np.array([[cos_val,-1*sin_val,0],[sin_val,cos_val,0],
                            [-1*c_x*cos_val-c_y*sin_val,c_x*sin_val-c_y*cos_val,1]])
    base_A_mat=np.array([[b*b,0,0],[0,a*a,0],[0,0,-1*a*a*b*b]])
    transformed_A_mat=(transform_mat.dot(base_A_mat)).dot(transform_mat.T)
    return transformed_A_mat



A=[0,0,4,2,0]
B=[4,-4.5,4,2,0.5*math.pi]
mat_A=para2mat(A)
#mat_A=mat_A/mat_A[1,1]
mat_B=para2mat(B)
#mat_B=mat_B/mat_B[1,1]

mat_A1,mat_A2,mat_A3=mat_A[:,0].reshape((3,1)),mat_A[:,1].reshape((3,1)),mat_A[:,2].reshape((3,1))
mat_B1,mat_B2,mat_B3=mat_B[:,0].reshape((3,1)),mat_B[:,1].reshape((3,1)),mat_B[:,2].reshape((3,1))
temp_alpaha_mat=mat_A
temp_alpaha=det(temp_alpaha_mat)

temp_beta=det(np.concatenate((mat_A1,mat_A2,mat_B3),axis=1))\
        +det(np.concatenate((mat_A1,mat_B2,mat_A3),axis=1))\
        +det(np.concatenate((mat_B1,mat_A2,mat_A3),axis=1))
#temp_beta=det(temp_beta_mat)

temp_gamma=det(np.concatenate((mat_A1,mat_B2,mat_B3),axis=1))\
        +det(np.concatenate((mat_B1,mat_A2,mat_B3),axis=1))\
        +det(np.concatenate((mat_B1,mat_B2,mat_A3),axis=1))
#temp_gamma=det(temp_gamma_mat)

temp_delta_mat=mat_B
temp_delta=det(temp_delta_mat)

temp_w=complex(-0.5,(3/4)**0.5)


temp_W=(-2)*pow(temp_beta,3)+9*temp_alpaha*temp_beta*temp_gamma\
    -27*pow(temp_alpaha,2)*temp_delta


temp_D=(-1)*pow(temp_beta,2)*pow(temp_gamma,2)+4*temp_alpaha*pow(temp_gamma,3)\
    +4*pow(temp_beta,3)*temp_delta-18*temp_alpaha*temp_beta*temp_gamma*temp_delta\
    +27*pow(temp_alpaha,2)*pow(temp_delta,3)

temp_val=round(27*temp_D,10)
str_temp=str(temp_val)
s=float(str_temp)
q=s**0.5

temp_Q=temp_W-temp_alpaha*pow(s,0.5)

temp_R=pow(4*temp_Q,1/3)

# temp_val1=2*np.dot(temp_beta,temp_beta)-6*np.dot(temp_alpaha,temp_gamma)
# temp_val2=(-1)*temp_beta
# temp_L=np.concatenate((temp_val1,temp_val2,temp_R),axis=1)
# temp_L=temp_L.T
L_1=2*pow(temp_beta,2)-6*temp_alpaha*temp_gamma
L_2=(-1)*temp_beta
L_3=temp_R
temp_L=np.array([[L_1],[L_2],[L_3]])

temp_M=3*temp_alpaha*np.array([[temp_R],[1],[2]])


temp_w_mat=np.array([[temp_w,1,temp_w**2],[1,1,1],[temp_w**2,1,temp_w]],dtype=complex)

# print(temp_L)
# print(temp_M)
# print(temp_w_mat)

temp_lambda=np.dot(temp_w_mat,temp_L)
temp_omega=np.dot(temp_w_mat,temp_M)

for i in range(3):
    print(temp_lambda[i],temp_omega[i])
    print("\n")


#mat_C=temp_lambda[0]*mat_A+temp_omega[0]*mat_B
#print(mat_C)
