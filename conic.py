import numpy as np
import math
from util import caculate_iou,para2mat,point_at_conic,conic_point
#from cacculate_conic_iou import caculate_iou,para2mat
from intersectConic_function import intersectConic
import matplotlib.pyplot as plt
np.set_printoptions(precision=4)


I_list=[]
A=[0,0,4,2,0]
z=[]
for i in range(0,100):
    z.append(i)

for i in range(0,100):



    B=[0,0,1,1,i*0.02*math.pi]
    mat_A=para2mat(A)
    mat_B=para2mat(B)

    if np.allclose(mat_A,mat_B):
    #如果两个二次型相同,直接输出iou=1
        I1=1
        I_list.append(I1)
        print(I1)
        continue


    else:
        p,flag=intersectConic(mat_A,mat_B)
        if flag==1:
            x=p[0,:]/p[2,:]
            y=p[1,:]/p[2,:]
            x=x.reshape((1,-1))
            y=y.reshape((1, -1))
            point=np.append(x,y,axis=0)
            point=point.T
            I1=caculate_iou(A,B,point,1)
            I1=I1.real
            #I2=caculate_iou(A,B,point,0)
            #I=(I1+I2)/2
            print(i)
        else:

            B_center=conic_point(B)
            A_center=conic_point(A)
            flag_1=point_at_conic(B_center, A, 1)
            flag_2=point_at_conic(A_center, B, 1)
            if flag_1 or flag_2:
                if flag_1:
                    iou=(B[2]*B[3])/(A[2]*A[3])
                else:
                    iou = (A[2] * A[3]) / (B[2] * B[3])
            else:
                iou=0
            I1=iou



    I_list.append(I1)
    print(I1)
plt.plot(z,I_list)
plt.show()





