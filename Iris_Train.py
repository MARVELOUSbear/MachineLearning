import svmutil as svm
from Iris import *
import matplotlib.pyplot as plt
import numpy as np

def Train(iris, traing_param_set='-t 2 -g 0.43'):
    TrainData=iris.GetTrainData()
    TrainLable=iris.GetTrainLabel()
    problem=svm.svm_problem(TrainLable.tolist(), TrainData.tolist())
    param=svm.svm_parameter(traing_param_set)
    m=svm.svm_train(problem, param)
    return m

def Predict(iris, mode):
    TestData  = iris.GetTestData().tolist()
    TestLabel = iris.GetTestLabel().tolist()   
    p_label, p_acc, p_val = svm.svm_predict(TestLabel, TestData, mode)
    return [TestLabel, p_label, p_acc, p_val]

if __name__ == "__main__":
    iris = Iris()
    iris.Open() 
    mode = Train(iris)
    a,b,c,d = Predict(iris, mode)
    a=np.array(a)
    b=np.array(b)
    num=np.zeros((3,3))
    for i in range(0,len(a)):
        num[int(a[i])-1][int(b[i])-1]+=1
    print(num)
    l=np.zeros((a.shape))
    for i in range(0,len(a)):
        for j in range(0,3):
            for k in range(0,3):
                if int(a[i])-1==j and int(b[i])-1==k:
                    l[i]=num[j][k]
    l=l*500
    plt.scatter(a,b,s=l,color="blue",alpha=0.4)
    
    