import os
import sys
import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt

def readData(DATA_PATH,DATA_SIZE,DATA_DIMENSION,test_rate):
    test_Labels = random.sample(range(0,DATA_SIZE-1),int(DATA_SIZE*test_rate))
    test_Data = [[0 for i in range(DATA_DIMENSION+1)] for j in range(len(test_Labels))]
    train_Data = [[0 for i in range(DATA_DIMENSION+1)] for j in range(DATA_SIZE-len(test_Labels))]
    #随机提取训练、测试数据
    f = open(DATA_PATH)
    line = f.readline()
    i = 0
    train_Label = 0
    test_Label = 0
    while line:
        temp = line.split(",",DATA_DIMENSION)
        if i in test_Labels:
            test_Data[test_Label][0:DATA_DIMENSION] = list(map(float, temp[0:DATA_DIMENSION]))
            test_Data[test_Label][DATA_DIMENSION] = temp[DATA_DIMENSION]
            test_Label = test_Label+1
        else:
            train_Data[train_Label][0:DATA_DIMENSION] = list(map(float, temp[0:DATA_DIMENSION]))
            train_Data[train_Label][DATA_DIMENSION] = temp[DATA_DIMENSION]
            train_Label = train_Label+1
        i = i+1
        line = f.readline()
    #将要训练的数据分类保存
    _1_Num = _2_Num = _3_Num = 0
    for i in range(len(train_Data)):
        if train_Data[i][DATA_DIMENSION] == 'Iris-setosa\n':
            _1_Num = _1_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-versicolor\n':
            _2_Num = _2_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-virginica\n':
            _3_Num = _3_Num+1
    _1_Data = mat(zeros((_1_Num,DATA_DIMENSION)))
    _2_Data = mat(zeros((_2_Num,DATA_DIMENSION)))
    _3_Data = mat(zeros((_3_Num,DATA_DIMENSION)))
    _1_Num = _2_Num = _3_Num = 0
    for i in range(len(train_Data)):
        if train_Data[i][DATA_DIMENSION] == 'Iris-setosa\n':
            _1_Data[_1_Num] = train_Data[i][0:DATA_DIMENSION]
            _1_Num = _1_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-versicolor\n':
            _2_Data[_2_Num] = train_Data[i][0:DATA_DIMENSION]
            _2_Num = _2_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-virginica\n':
            _3_Data[_3_Num] = train_Data[i][0:DATA_DIMENSION]
            _3_Num = _3_Num+1
    f.close()
    return train_Data,test_Data,_1_Data.T,_2_Data.T,_3_Data.T

def calcu_Eucli_Dis(x,y):
    dis = np.sqrt(np.sum(np.square(x-y)))
    return dis

def Nearest_Neighbors(dataPath,dataSize,dataDimension,dataTypeNum,testRate = 2/5):
    #数据准备
    train_Data,test_Data,group1,group2,group3 = readData(dataPath,dataSize,dataDimension,testRate)
    print("*********************************************************************")
    print("                    train_Size:",len(train_Data))
    print("                    test_Size:",len(test_Data))
    print("*********************************************************************")
    
    wrongNum = correctNum = 0

    #遍历每一个测试元素
    for testElement in range(len(test_Data)):
        minDis = 1024
        minLabel = -1
        print("**********************testElement_",testElement,"************************")
        #每一个测试元素与训练集中的每一个数据进行距离计算
        for trainElement in range(len(train_Data)):
            testArray = np.array(test_Data[testElement][0:dataDimension])
            trainArray = np.array(train_Data[trainElement][0:dataDimension])
            #计算距离
            distance = calcu_Eucli_Dis(testArray,trainArray)
            #寻找最近邻元素
            if distance < minDis:
                minDis = distance
                minLabel = trainElement

        print("minDis:",minDis)
        print("Real:",test_Data[testElement][dataDimension],"Forecast:",train_Data[minLabel][dataDimension])

        if(test_Data[testElement][dataDimension] == train_Data[minLabel][dataDimension]):
            correctNum += 1
        else:
            wrongNum += 1
    print("------------------------------------")
    print("|WrongNums:",wrongNum,"correctNums:",correctNum)
    print("------------------------------------\n\n\n")
    Accuracy = correctNum/(correctNum+wrongNum)
    return  Accuracy

def dis_Sort(allDisInfo):
    for i in range(0,len(allDisInfo)): 
        for j in range(0,len(allDisInfo)-i-1):
            if allDisInfo[j][1]>allDisInfo[j+1][1]:
                temp = allDisInfo[j]
                allDisInfo[j] = allDisInfo[j+1]
                allDisInfo[j+1] = temp
    return allDisInfo

def k_Judge(allDisInfo,k):
    type_1 = type_2 = type_3 = 0
    for i in range(k):
        if allDisInfo[i][0] == "Iris-setosa\n":
            type_1 += 1
        if allDisInfo[i][0] == "Iris-versicolor\n":
            type_2 += 1
        if allDisInfo[i][0] == "Iris-virginica\n":
            type_3 += 1
    if type_1>=type_2 and type_1>=type_3:
        return "Iris-setosa\n"
    if type_2>=type_1 and type_2>=type_3:
        return "Iris-versicolor\n"
    if type_3>=type_1 and type_3>=type_2:
        return "Iris-virginica\n"

def K_Nearest_Neighbors(dataPath,dataSize,dataDimension,dataTypeNum,testRate,k=5):
    #数据准备
    train_Data,test_Data,group1,group2,group3 = readData(dataPath,dataSize,dataDimension,testRate)
    print("*********************************************************************")
    print("                    train_Size:",len(train_Data))
    print("                    test_Size:",len(test_Data))
    print("*********************************************************************")
    
    wrongNum = correctNum = 0
    #遍历每一个测试元素
    for testElement in range(len(test_Data)):
        allDisInfo = [[0 for i in range(2)] for j in range(len(train_Data))]  #[forcast_label][distance]

        print("**********************testElement_",testElement,"************************")
        #每一个测试元素与训练集中的每一个数据进行距离计算
        for trainElement in range(len(train_Data)):
            testArray = np.array(test_Data[testElement][0:dataDimension])
            trainArray = np.array(train_Data[trainElement][0:dataDimension])
            #计算距离
            dis = calcu_Eucli_Dis(testArray,trainArray)
            #保存所有label的距离信息
            allDisInfo[trainElement][0] = train_Data[trainElement][dataDimension]
            allDisInfo[trainElement][1] = dis
        
        #对距离信息进行排序
        dis_Sort(allDisInfo)
        for i in range(len(allDisInfo)):
            print(allDisInfo[i])
        #选择K值进行判别
        forcast_label = k_Judge(allDisInfo,k)
        print("------------------------------------> Forcast Label:",forcast_label,"-----------------------------------> Real Label:",test_Data[testElement][dataDimension])
        if(test_Data[testElement][dataDimension] == forcast_label):
            correctNum += 1
        else:
            wrongNum += 1

    print("------------------------------------")
    print("|WrongNums:",wrongNum,"correctNums:",correctNum)
    print("------------------------------------\n\n\n")
    Accuracy = correctNum/(correctNum+wrongNum)
    return  Accuracy


if __name__ == "__main__":
    Accuracy_AVE = 0
    Accuracy = []
    Accuracy.append(0)
    testTimes = 20
    for i in range(testTimes):
        temp = K_Nearest_Neighbors("iris.data",150,4,3,1/5,5)
        # temp = Nearest_Neighbors("iris.data",150,4,3,1/5)
        Accuracy.append(temp)
        plt.scatter(i+1,temp)
        Accuracy_AVE += temp
    Accuracy_AVE /= testTimes
    
    for i in range(testTimes):
        print("Accuracy_",i,": ",Accuracy[i+1])
    print("Accuracy_AVE:",Accuracy_AVE)
    plt.title("Accuracy")
    plt.xlabel("testTimes")
    plt.ylabel("accuracy")
    plt.ylim(0,2)
    plt.plot(Accuracy)

    plt.show()