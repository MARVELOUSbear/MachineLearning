# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:19:42 2017

@author: Jarily
"""
# coding: UTF-8
import numpy as np
from sklearn.model_selection import train_test_split
from AdaBoost import AdaBoost
  
def main():
    dataset = np.loadtxt('pima-indians-diabetes.data', delimiter=",")
    X=dataset[:,0:7]
    y_tmp=dataset[:,8]
    y=np.ones(y_tmp.shape)
    y[y_tmp==1]=1
    y[y_tmp==0]=-1
    X1,X2,y1,y2=train_test_split(X, y, test_size=0.33, random_state=2)
    X1=list(X1.transpose())
    X2=list(X2.transpose())
    ada=AdaBoost(X1,y1)
    ada.train(10)
    result=ada.pred(X2)
    result=result.tolist()
    
    y2=y2.tolist()
    cnt=0
    sum=0
    for i in range (len(result)):
        if y2[i]==result[i]:
            cnt+=1
        sum+=1
        
    print("测试样本总数：",sum)
    print("测试正确样本数：",cnt)
    print("正确率为：",(float(cnt)/sum))

if __name__=='__main__':
    main()