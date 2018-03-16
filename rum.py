# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:29:37 2016
Run Kmeans classifier
@author: liudiwei
"""
import numpy as np
from kmeans import KMeansClassifier
import matplotlib.pyplot as plt

def chooseK(data_X,data_Y):
    best_rd=0
    best_k=0
    for k in range(1,100):
        clf = KMeansClassifier(k)
        clf.fit(data_X)
        labels = clf._labels
        a=0
        b=0
        c=0
        d=0
        for j in range(len(data_Y)-1,0,-1):
            for i in range(0,j):
                if data_Y[i]==data_Y[j] and labels[i]==labels[j]:
                    a+=1
                elif data_Y[i]==data_Y[j] and labels[i]!=labels[j]:
                    b+=1
                elif data_Y[i]!=data_Y[j] and labels[i]==labels[j]:
                    c+=1
                else:
                    d+=1
        rd=2*(a+d)/(len(data_Y)*(len(data_Y)-1))
        print("rd ",rd,"k ",k)
        if rd>best_rd:
            best_rd=rd
            best_k=k
    return best_rd,best_k

if __name__=="__main__":
    data_X =np.loadtxt("data/wine.data",delimiter=",",usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
    data_Y=np.loadtxt("data/wine.data",delimiter=",",usecols=(0))

    rd,k=chooseK(data_X,data_Y)
    print("best rd: ",rd,"best k: ",k)
    clf = KMeansClassifier(k)
    clf.fit(data_X)
    cents = clf._centroids
    labels = clf._labels
    sse = clf._sse
    
    colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']
    for i in range(k):
        index = np.nonzero(labels==i)[0]
        x0 = data_X[index, 0]
        x1 = data_X[index, 1]
        y_i = i
        
        for j in range(len(x0)):
            plt.text(x0[j], x1[j], str(y_i),color=colors[i],)
        plt.scatter(cents[i,0],cents[i,1],marker='x',linewidths=7)
    outname = "./result/k_clusters" + str(k) + ".png"
    plt.savefig(outname)
    plt.show()
    
    