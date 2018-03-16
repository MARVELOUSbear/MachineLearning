# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:09:15 2016
K-means cluster
@author: liudiwei
"""

import numpy as np

class KMeansClassifier():

    def __init__(self, k=3, initCent='random', max_iter=500 ):
        
        self._k = k
        self._initCent = initCent
        self._max_iter = max_iter
        self._clusterAssment = None
        self._labels = None
        self._sse = None
        
    def _calEDist(self, arrA, arrB):#欧式距离计算
        return np.math.sqrt(sum(np.power(arrA-arrB,2)))
    
    def _randCent(self, data_X, k):#随机选取k个质心
        n=data_X.shape[1] #获取特征的维数
        centroids=np.zeros((k,n))  #生成k*n的矩阵用于存储质心
        for j in range(0,n):
            min_j=min(data_X[:, j])
            range_j=float(max(data_X[:, j]-min_j))
            centroids[:,j]=(min_j+range_j*np.random.rand(k,1)).flatten()
        return centroids 
    
    def fit(self, data_X):
        #类型检查
        if not isinstance(data_X, np.ndarray) \
           or isinstance(data_X, np.matrixlib.defmatrix.matrix):
            try:
                data_X=np.asarray(data_X)
            except:
                raise TypeError("numpy.ndarray resuired for data_X")         
        m=data_X.shape[0]  #获取样本的个数
        #一个m*2的二维矩阵
        #矩阵第一列存储样本点所属的族的索引值
        #第二列存储该点与所属族的质心的平方误差
        self._clusterAssment=np.zeros((m,2)) 
        if self._initCent=='random':
            self._centroids=self._randCent(data_X, self._k)   
        clusterChanged=True
        for out_iter in range(self._max_iter):
            clusterChanged=False
            for i in range(m):   #将每个样本点分配到离它最近的质心所属的族
                min_dist=np.inf #首先将minDist置为一个无穷大的数
                min_index=-1    #将最近质心的下标置为-1
                for j in range(self._k): #次迭代用于寻找最近的质心
                    arrA=self._centroids[j,:]
                    arrB=data_X[i,:]
                    dist_ji=self._calEDist(arrA, arrB) #计算误差值
                    if dist_ji < min_dist:
                        min_dist = dist_ji
                        min_index = j
                if self._clusterAssment[i,0] !=min_index:
                    clusterChanged = True
                    self._clusterAssment[i,:] = min_index, min_dist**2
            if not clusterChanged:#若所有样本点所属的族都不改变,则已收敛,结束迭代
                break
            for i in range(self._k):#更新质心，将每个族中的点的均值作为质心
                index_all = self._clusterAssment[:,0] #取出样本所属簇的索引值
                value = np.nonzero(index_all==i) #取出所有属于第i个簇的索引值
                ptsInClust = data_X[value[0]]    #取出属于第i个簇的所有样本点
                self._centroids[i,:] = np.mean(ptsInClust, axis=0) #计算均值
        
        self._labels = self._clusterAssment[:,0]
        self._sse = sum(self._clusterAssment[:,1])
    
    def predict(self, X):
        if not isinstance(X,np.ndarray):
            try:
                X = np.asarray(X)
            except:
                raise TypeError("numpy.ndarray required for X")   
        m = X.shape[0]
        preds = np.empty((m,))
        for i in range(m):#将每个样本点分配到离它最近的质心所属的族
            minDist = np.inf
            for j in range(self._k):
                distJI = self._calEDist(self._centroids[j,:], X[i,:])
                if distJI < minDist:
                    minDist = distJI
                    preds[i] = j
        return preds

        