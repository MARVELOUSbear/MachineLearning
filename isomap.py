import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

filename='pima-indians-diabetes.data' #file directory


def loadFile():   #read the .data file of the dataset
    fr = open(filename)
    linestr = fr.readlines()
    dataMat = []
    length=len(linestr)
    for i in range(0,length-1): #100 data samples->leave 10 samples out and append the rest samples
        lineArr = linestr[i].strip().split(",")
        w2=float(lineArr[0])
        w3=float(lineArr[1])
        w4=float(lineArr[2])
        w5=float(lineArr[3])
        dataMat.append([w2, w3,w4,w5])
    return np.array(dataMat)

def classify(inputPoint,dataSet,k):
    dataSetSize = dataSet.shape[0]     #已知分类的数据集（训练集）的行数
    #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
    diffMat = np.tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
    sqDiffMat = diffMat ** 2                    #差值矩阵平方
    sqDistances = sqDiffMat.sum(axis=1)         #计算每一行上元素的和
    distances = sqDistances ** 0.5              #开方得到欧拉距离矩阵
    sortedDistIndicies = distances.argsort()    #按distances中元素进行升序排序后得到的对应下标的列表
    #选择距离最小的k个点
    for i in range(k+1,dataSetSize):
        distances[sortedDistIndicies[i]]=float("inf")
    return distances

def createDist(k):
    dataset = loadFile()
    dataSetSize = dataset.shape[0]
    totaldist=np.zeros((dataSetSize,dataSetSize))
    resultlist=np.zeros((dataSetSize,dataSetSize))
    G = nx.Graph()
    for i in range(0,dataSetSize):
        totaldist[i]=classify(dataset[i],dataset,k)
    for i in range(0,dataSetSize):
        for j in range(0,dataSetSize):
            if totaldist[i][j]!=float("inf"):
                elist=[(i,j,totaldist[i][j])]
                G.add_weighted_edges_from(elist)
    for i in range(0,dataSetSize):
        for j in range(0,dataSetSize):
                if nx.has_path(G,i,j):
                    length=nx.dijkstra_path_length(G,i,j)
                    resultlist[i][j]=length
                else:
                    resultlist[i][j]=float("inf")
    return resultlist

def mds(D,q):
    D = np.asarray(D)
    DSquare = D**2
    totalMean = np.mean(DSquare)
    columnMean = np.mean(DSquare, axis = 0)
    rowMean = np.mean(DSquare, axis = 1)
    B = np.zeros(DSquare.shape)
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            B[i][j] = -0.5*(DSquare[i][j] - rowMean[i] - columnMean[j]+totalMean)
    eigVal,eigVec = np.linalg.eig(B)
    X = np.mat(np.dot(eigVec[:,:q],np.sqrt(np.diag(eigVal[:q]))),dtype="float")
    return X

def plotData(X):
    plt.plot(X[:,0],X[:,1],'o')
    plt.show()

def main():
    X=mds(createDist(51),2)
    plotData(X)
    
if __name__ == "__main__" :
    main()