import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

FILE_PATH = r'./Data/iris.data'


class Iris(object):
    __LABEL_SETOSA      = 1
    __LABEL_VERSICOLOUR = 2
    __LABEL_VIRGINICA   = 3
    
    def __init__(self):
        pass
    
    def __LoadFile(self, dataPath):
        raw_data = np.loadtxt(FILE_PATH,delimiter=',',usecols=(0, 1, 2, 3),dtype=float)
        classes = {b'Iris-setosa'    : Iris.__LABEL_SETOSA,
                   b'Iris-versicolor': Iris.__LABEL_VERSICOLOUR,
                   b'Iris-virginica' : Iris.__LABEL_VIRGINICA}
        label = np.loadtxt(FILE_PATH,delimiter=',',
                           converters={4: lambda x: classes[x]},usecols=(4))
        return [raw_data, label]
    
    def __SplitData(self, d, l):
        traindata, d_test, trainlabel, testdata = train_test_split(d, l, test_size=0.4)  
        return [traindata, d_test, trainlabel, testdata]

    def Open(self, dataPath=FILE_PATH):
        d,l = self.__LoadFile(dataPath)
        traindata, testdata, trainlabel, testlabel = self.__SplitData(d, l)
        self.traindata=traindata
        self.testdata=testdata
        self.trainlabel=trainlabel
        self.testlabel=testlabel
        self._index_in_epoch = 0
        self._num_samples, _ = self.traindata.shape
    
    def GetTrainData(self):
        return self.traindata
    def GetTrainLabel(self):
        return self.trainlabel
    def GetTestData(self):
        return self.testdata
    def GetTestLabel(self):
        return self.testlabel
    
if __name__ == "__main__":
    import os
    os.chdir("C:\Program Files\libsvm-3.22\python")
