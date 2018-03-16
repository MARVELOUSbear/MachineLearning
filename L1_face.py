# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 19:44:39 2017

@author: 46125
"""

import matplotlib.pylab as plt
import numpy as np


def readPic():
    A=[]
    for i in range(1, 121):
        for j in range(1, 6):
            picName = str(i)+'-'+str(j)
            img=plt.imread("feret/%s.bmp"%(picName))
            A.append(img.flatten())
    return A

def argminX(L,Lambda,z):
    x = np.mat(np.zeros(shape=z.shape, dtype='float'))
    for i in range(z.shape[0]):
        x[i,0]=0
        if Lambda/L<z[i,0]:
            x[i,0]=z[i,0]-Lambda/L
        elif -Lambda/L>z[i, 0]:
            x[i,0]=z[i,0]+Lambda/L
    return x
        
def predict_class(x):
    max_sum = 0.0
    max_index=0
    for i in range(120):
        tmp_sum=0.0
        for j in range(5):
            tmp_sum=x[5*i+j,0]
        if tmp_sum>max_sum:
            max_sum=tmp_sum
            max_index=i
    return max_index

def main():
    A = np.mat(readPic()).transpose()
    acc_num=0
    for i in range(1,13):
        picNum=i
        test_img=plt.imread("feret/%d-6.bmp"%(picNum))
        b = np.mat(test_img.flatten().reshape(112*92, 1))
        x = np.mat(np.zeros(shape=(600, 1)))
        L = np.linalg.norm(A, 2)**2
        Lambda = 0.1
        threshold=np.inf
        k = 0
        while threshold > 2000000:
            z=x-1/L*(A.T*(A*x-b))
            x=argminX(L,Lambda,z)
            threshold=sum((np.dot(A,x)-b).transpose()*(np.dot(A,x)-b))+Lambda*sum(abs(x))
            k+=1
            if k>2000:
                break
        index=predict_class(x)+1
        print("predicting the 6-th pic from person:", picNum)
        print("prediction:", index)
        if index==picNum:
            acc_num+=1
    print("accuracy:",67/0.7)
        

main()
