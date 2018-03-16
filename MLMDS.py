# -*- coding: utf-8 -*-
"""
Machine Learning lab2
@author: 10152510119 徐紫琦
"""
import numpy as np
import matplotlib.pyplot as plt

filename='iris.data' 

def loadFile():   #read the .data file of the dataset
    fr = open(filename)
    linestr = fr.readlines()
    dataMat = []
    length=len(linestr)
    for i in range(0,length-1): 
        lineArr = linestr[i].strip().split(",")
        w2=float(lineArr[0])
        w3=float(lineArr[1])
        w4=float(lineArr[2])
        w5=float(lineArr[3])
        dataMat.append([w2, w3,w4,w5])
    return np.mat(dataMat)

def calculateDistance(dataMat):
    length=dataMat.shape[0]
    distance=np.mat(np.zeros((length,length)))
    for i in range(0,length):
        for j in range(0,length):
            distance[(i,j)]=np.linalg.norm(dataMat[i]-dataMat[j],2)
    return distance

def mds(D,q):
    D = np.asarray(D)
    #计算距离平方矩阵
    DSquare = D**2
    #计算总均值，列均值和行均值
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    #计算矩阵B中各元素的值
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5*(DSquare[i][j] - rowMean[i] - columnMean[j]+totalMean)
    #对矩阵B做特征值分解
    eigVal,eigVec = np.linalg.eig(B)
    #得到降维以后的坐标
    X = np.mat(np.dot(eigVec[:,:q],np.sqrt(np.diag(eigVal[:q]))),dtype="float")
    return X

def plotData(X):
    plt.plot(X[:,0],X[:,1],'o')
    plt.show()

def main():
    file=loadFile()
    dist=calculateDistance(file)
    X=mds(dist,2)
    plotData(X)

if __name__=='__main__':
    main()