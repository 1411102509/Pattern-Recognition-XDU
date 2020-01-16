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
        temp = line.split(",",4)
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
        if train_Data[i][4] == 'Iris-setosa\n':
            _1_Num = _1_Num+1
        if train_Data[i][4] == 'Iris-versicolor\n':
            _2_Num = _2_Num+1
        if train_Data[i][4] == 'Iris-virginica\n':
            _3_Num = _3_Num+1
    _1_Data = mat(zeros((_1_Num,DATA_DIMENSION)))
    _2_Data = mat(zeros((_2_Num,DATA_DIMENSION)))
    _3_Data = mat(zeros((_3_Num,DATA_DIMENSION)))
    _1_Num = _2_Num = _3_Num = 0
    for i in range(len(train_Data)):
        if train_Data[i][4] == 'Iris-setosa\n':
            _1_Data[_1_Num] = train_Data[i][0:4]
            _1_Num = _1_Num+1
        if train_Data[i][4] == 'Iris-versicolor\n':
            _2_Data[_2_Num] = train_Data[i][0:4]
            _2_Num = _2_Num+1
        if train_Data[i][4] == 'Iris-virginica\n':
            _3_Data[_3_Num] = train_Data[i][0:4]
            _3_Num = _3_Num+1
    f.close()
    return train_Data,test_Data,_1_Data.T,_2_Data.T,_3_Data.T

#计算样本均值 
def compute_mean(samples):
	mean_mat=mean(samples, axis=1)
	return mean_mat

#计算样本类内离散度
#参数samples表示样本向量矩阵，大小为nxm，其中n表示维数，m表示样本个数
#参数mean表示均值向量，大小为1xd，d表示维数，大小与样本维数相同，即d=m
def compute_withinclass_scatter(samples, mean):
    #获取样本维数，样本个数	
    dimens,nums=samples.shape
    #将所有样本向量减去均值向量
    samples_mean=samples-mean
    #初始化类内离散度矩阵	
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
    s_w_12 = s_in1+s_in2
    s_w_13 = s_in1+s_in3
    s_w_23 = s_in2+s_in3

    #求解权向量
    w_12 = dot(s_w_12.I,mean1-mean2)
    w_13 = dot(s_w_13.I,mean1-mean3)
    w_23 = dot(s_w_23.I,mean2-mean3)

    #求解阈值
    w0_12 = -0.5*(dot(w_12.T,mean1)+dot(w_12.T,mean2))
    w0_13 = -0.5*(dot(w_13.T,mean1)+dot(w_13.T,mean3))
    w0_23 = -0.5*(dot(w_23.T,mean2)+dot(w_23.T,mean3))

    #显示训练结果
    showTrainResule(group1,group2,w_12,w0_12)
    showTrainResule(group1,group3,w_12,w0_13)
    showTrainResule(group2,group3,w_12,w0_23)
    
    #测试结果
    for i in range(len(test_Data)):
        test1 = mat(zeros((1,4)))
        test1[0] = test_Data[i][0:4]
        g_12 = dot(w_12.T,test1.T)+w0_12
        g_13 = dot(w_13.T,test1.T)+w0_13
        g_23 = dot(w_23.T,test1.T)+w0_23
        if test_Data[i][4] == 'Iris-setosa\n':
            color = 'b'
            if g_12>0 and g_13>0:
                correctNum += 1
            else:
                wrongNum += 1
        if test_Data[i][4] == 'Iris-versicolor\n':
            color = 'm'
            if g_12<0 and g_23>0:
                correctNum += 1
            else:
                wrongNum += 1
        if test_Data[i][4] == 'Iris-virginica\n':
            color = 'c'
            if g_13<0 and g_23<0:
                correctNum += 1
            else:
                wrongNum += 1
        # plt.scatter(float(g_12),3,20,color)
        # plt.scatter(float(g_13),2,20,color)
        # plt.scatter(float(g_23),1,20,color)
    Accuracy = correctNum/(correctNum+wrongNum)
    # plt.title("Test Result")
    # plt.show()
    return Accuracy


if __name__ == "__main__":
    Accuracy_AVE = 0
    Accuracy = []
    Accuracy.append(0)
    testTimes = 20
    for i in range(testTimes):
        temp = LDA_Fisher("iris.data",150,4,3,2/5)
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


