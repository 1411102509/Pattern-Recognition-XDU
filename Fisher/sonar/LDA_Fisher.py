import os
import sys
import numpy as np
from numpy import *
import random
import matplotlib.pyplot as plt

def readData(DATA_PATH,DATA_SIZE,DATA_DIMENSION,test_rate):
    test_Labels = random.sample(range(0,DATA_SIZE-1),int(DATA_SIZE*test_rate))
    test_Data = [[0 for i in range(5)] for j in range(len(test_Labels))]
    train_Data = [[0 for i in range(5)] for j in range(DATA_SIZE-len(test_Labels))]
    #随机提取训练、测试数据
    f = open(DATA_PATH)
    line = f.readline()
    i = 0
    train_Label = 0
    test_Label = 0
    while line:
        temp = line.split(",",DATA_DIMENSION)
        if i in test_Labels:
            test_Data[test_Label] = temp
            test_Label = test_Label+1
        else:
            train_Data[train_Label] = temp
            train_Label = train_Label+1
        i = i+1
        line = f.readline()
    #将要训练的数据分类保存
    _1_Num = _2_Num = _3_Num = 0
    for i in range(len(train_Data)):
        if train_Data[i][DATA_DIMENSION] == 'R\n':
            _1_Num = _1_Num+1
        if train_Data[i][DATA_DIMENSION] == 'M\n':
            _2_Num = _2_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-virginica\n':
            _3_Num = _3_Num+1
    _1_Data = mat(zeros((_1_Num,DATA_DIMENSION)))
    _2_Data = mat(zeros((_2_Num,DATA_DIMENSION)))
    _3_Data = mat(zeros((_3_Num,DATA_DIMENSION)))
    _1_Num = _2_Num = _3_Num = 0
    for i in range(len(train_Data)):
        if train_Data[i][DATA_DIMENSION] == 'R\n':
            _1_Data[_1_Num] = train_Data[i][0:DATA_DIMENSION]
            _1_Num = _1_Num+1
        if train_Data[i][DATA_DIMENSION] == 'M\n':
            _2_Data[_2_Num] = train_Data[i][0:DATA_DIMENSION]
            _2_Num = _2_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-virginica\n':
            _3_Data[_3_Num] = train_Data[i][0:DATA_DIMENSION]
            _3_Num = _3_Num+1
    f.close()
    return train_Data,test_Data,_1_Data.T,_2_Data.T,_3_Data.T

#计算样本均值 
def compute_mean(samples):
	mean_mat=mean(samples, axis=1)
	return mean_mat

#计算样本类内离散度
def compute_withinclass_scatter(samples, mean):
    #获取样本维数，样本个数	
    dimens,nums=samples.shape
    #将所有样本向量减去均值向量
    samples_mean=samples-mean
    s_in=0	
    for i in range(nums):
        x=samples_mean[:,i]
        s_in+=dot(x,x.T)
    return s_in

def showTrainResule(group1,group2,w,w0):
    dimens,nums=group1.shape
    for i in range(nums):
        position = dot(w.T,group1[:,i])+w0
        plt.scatter(float(position),0,20,'m','x')
    dimens,nums=group2.shape
    for i in range(nums):
        position = dot(w.T,group2[:,i])+w0
        plt.scatter(float(position),0,20,'c','x')
    plt.show()

def LDA_Fisher(dataPath,dataSize,dataDimension,dataTypeNum,testRate = 2/5):
    Accuracy = 0.0
    correctNum = wrongNum = 0
    train_Data,test_Data,group1,group2,group3 = readData(dataPath,dataSize,dataDimension,testRate)

    #求均值向量
    mean1 = compute_mean(group1)
    mean2 = compute_mean(group2)
    mean3 = compute_mean(group3)

    #求类内离散度
    s_in1 = compute_withinclass_scatter(group1, mean1)
    s_in2 = compute_withinclass_scatter(group2, mean2)
    s_in3 = compute_withinclass_scatter(group3, mean3)

    #求总类内离散度矩阵
    s_w = s_in1+s_in2

    #求解权向量
    w = dot(s_w.I,mean1-mean2)

    #求解阈值
    w0 = -0.5*(dot(w.T,mean1)+dot(w.T,mean2))

    #绘图，训练样本投影
    # showTrainResule(group1,group2,w,w0)

    #绘图，测试样本投影
    for i in range(len(test_Data)):
        test1 = mat(zeros((1,dataDimension)))
        test1[0] = test_Data[i][0:dataDimension]
        g = dot(w.T,test1.T)+w0
        if test_Data[i][dataDimension] == 'R\n':
            if g>0:
                correctNum += 1
            else:
                wrongNum += 1
            # plt.scatter(float(g),1,20,'m')
        if test_Data[i][dataDimension] == 'M\n':
            if g<0:
                correctNum += 1
            else:
                wrongNum += 1
            # plt.scatter(float(g),-1,20,'c')

    Accuracy = correctNum/(correctNum+wrongNum)
    # plt.show()
    return Accuracy
    


if __name__ == "__main__":
    Accuracy_AVE = 0
    Accuracy = []
    Accuracy.append(0)
    testTimes = 20
    for i in range(testTimes):
        temp = LDA_Fisher("sonar.all-data",208,60,2,2/5)
        Accuracy.append(temp)
        print(i+1,":Accuracy:",temp)
        plt.scatter(i+1,temp)
        Accuracy_AVE += temp
    Accuracy_AVE /= testTimes
    print("Accuracy_AVE:",Accuracy_AVE)
    plt.title("Accuracy")
    plt.xlabel("testTimes")
    plt.ylabel("accuracy")
    plt.ylim(0,2)
    plt.plot(Accuracy)

    plt.show()