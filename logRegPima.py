# -*- coding: utf-8 -*-
"""
Machine Learning lab1
comparison with the Pima dataset
uesed 760 samples in total
each two validation method has 76 validation data sample
@author: 10152510119 徐紫琦
"""

from numpy import *
import scipy
import matplotlib.pyplot as plt

filename='pima-indians-diabetes.data' #file directory

def loadFile():   #read the .data file of the dataset
    fr = open(filename)
    linestr = fr.readlines()
    return linestr

def appendData(linestr,dataMat,labelMat,i):#append attributes into the data matrix and append class label into the label matrix
    lineArr = linestr[i].strip().split(",")#turn the raw string into a list
    for i in range(0,8):
        w2=float(lineArr[0])
        w3=float(lineArr[1])
        w4=float(lineArr[2])
        w5=float(lineArr[3])
        w6=float(lineArr[4])
        w7=float(lineArr[5])
        w8=float(lineArr[6])
        w9=float(lineArr[7])
    dataMat.append([1.0, w2, w3,w4,w5,w6, w7,w8,w9])# 8 attributes given, w1=1
    labelMat.append([int(lineArr[8])])

def loadDataSet10(linestr,j):  #turn raw data into matrixes for the j-th computation in 10-fold cross-validation
    dataMat = []
    labelMat = []
    for i in range(0,76*j): #760 data samples->leave 76 samples out and append the rest samples
        appendData(linestr,dataMat,labelMat,i)
    if(j<9):    
        for i in range(76*(j+1),760):
            appendData(linestr,dataMat,labelMat,i)
    return dataMat,labelMat


def loadDataSet1(linestr,j):   #turn raw data into matrixes for the j-th computation in leave-one-out cross-validation
    dataMat = []
    labelMat = []
    for i in range(0,j-1):  #760 data samples->leave 1 sample out and append the rest samples
        appendData(linestr,dataMat,labelMat,i)
    if(j<759):    
        for i in range(j+1,760):
            appendData(linestr,dataMat,labelMat,i)
    return dataMat,labelMat

def sigmoid(inX):  #sigmoid function: special exp function to prevent overflow
    return scipy.special.expit(-inX)

def gradAscent(dataMat, labelMat): 
    dataMatrix=mat(dataMat)
    alpha = 0.001  #learning rate
    maxCycles = 1000 #the number of circles of iteration
    weights = ones((9,1)) #initialize the parameters
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()* error #refresh the parameters
    return weights

def countErr10(weights,linestr,j):  #count the number of errors in the j-th computation in 10-fold cross-validation
    dataMat = []
    labelMat = []
    for i in range(76*(j),76*(j+1)):  #construct the validation data matrix
        appendData(linestr,dataMat,labelMat,i)   
    y=sigmoid(mat(dataMat)*weights)
    err=0
    for i in range(0,76): #compare the class labels with the pridiction values to count errors
        if (round(y[(i,0)])!=labelMat[i][0] or (round(y[(i,0)])==0.5 and labelMat[i][0]==1)): #to solve the problem that round(0.5,0)==0.0
            err+=1
    return err


def countErr1(weights,linestr,j):   #judge if the validation data is an error
    dataMat = []
    labelMat = []
    appendData(linestr,dataMat,labelMat,j)   
    y=sigmoid(mat(dataMat)*weights)
    err=0
    if (round(y[(0,0)])!=labelMat[0][0]) :
        err+=1
    return err

def printFigure(err10,err1):
    n_groups = 2
    list=[err10,err1]
    fig, ax = plt.subplots()  
    index = np.arange(n_groups)  
    bar_width = 0.35     
    opacity = 0.4  
    rects1 = plt.bar(index, list, bar_width,alpha=opacity, color='b')  
    plt.ylabel('error rate')  
    plt.title('Error rate comparison between 10-fold\n cross-validation and leave-one-out cross-validation')  
    plt.xticks(index, ('10-fold', 'leave-one-out'))  
    plt.ylim(0,1)  
    plt.legend()  
    plt.tight_layout()  
    plt.show()
    

def main():
    totalErr10=0
    totalErr1=0
    linestr=loadFile()
    for i in range(0,10): #run 10 times 10-fold cross-validation
        dataMat, labelMat = loadDataSet10(linestr,i)
        weights=gradAscent(dataMat, labelMat).getA()
        totalErr10+=countErr10(weights,loadFile(),i)
        print(totalErr10)
    for j in range(0,760):
        dataMat, labelMat = loadDataSet1(linestr,j)
        weights=gradAscent(dataMat, labelMat).getA()
        err=countErr1(weights,loadFile(),j)
        totalErr1+=err
    printFigure(totalErr10/760,totalErr1/760)
    print(totalErr10,totalErr1)

if __name__=='__main__':
    main()