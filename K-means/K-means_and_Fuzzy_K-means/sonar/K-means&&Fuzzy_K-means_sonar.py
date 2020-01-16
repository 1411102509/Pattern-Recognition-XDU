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
        if train_Data[i][DATA_DIMENSION] == 'R\n':
            _1_Num = _1_Num+1
        if train_Data[i][DATA_DIMENSION] == 'M\n':
            _2_Num = _2_Num+1
        if train_Data[i][DATA_DIMENSION] == 'Iris-virginica\n':
            _3_Num = _3_Num+1
    _1_Data = mat(np.zeros((_1_Num,DATA_DIMENSION)))
    _2_Data = mat(np.zeros((_2_Num,DATA_DIMENSION)))
    _3_Data = mat(np.zeros((_3_Num,DATA_DIMENSION)))
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

#计算欧氏距离
def calcu_Eucli_Dis(x,y):
    dis = np.sqrt(np.sum(np.square(np.array(x)-np.array(y))))
    return dis

#计算样本到均值向量的距离平方和
def calcu_error(samples, mean):
    sum=0	
    for i in range(len(samples)):
        e = np.sum(np.square(np.array(samples[i])-np.array(mean)))
        sum += e

    return sum


#*************************************************************K-means********************************************************************
#K-means类结构
class Data_K_means:
    def __init__(self): 
        self.name = ''

        self.clusteringCenter = []  #聚类中心
        self.clusteringData = []    #包含的样本数据
        self.clusteringMeans = 0    #均值
        self.clusteringError = 0    #每一类中的最小误差平方和

        self.R = 0
        self.M = 0

    def print(self):
        print("**************************   ",self.name,"   ***********************")

        print("clusteringCenter: ",self.clusteringCenter)
        print("clusteringError: "  ,self.clusteringError)
        # print("clusteringData: \n"  ,np.array(self.clusteringData))
        print("dataSize: "  ,len(self.clusteringData))

#按最小距离准则划分样本到N类
def divide_Sample(train_Data,category,dataDimension,K):
    for i in range(K):
        category[i].clusteringData = []
    for i in range(len(train_Data)):
        min_dis = 1024
        min_dis_label = -1
        for j in range(K):
            dis = calcu_Eucli_Dis(train_Data[i][:dataDimension],category[j].clusteringCenter)
            if dis<min_dis:
                min_dis = dis
                min_dis_label = j
        # print("min_dis:",min_dis,"label:",min_dis_label)
        category[min_dis_label].clusteringData.append(train_Data[i])

#更新参数：计算均值与准则函数，更新聚类中心,返回准则函数
def update_Cluster_Center(category,dataDimension,K):
    #计算每一类的均值、误差J_i,计算出准则函数
    J = 0
    for i in range(K):
        dataArray = np.array(np.array(category[i].clusteringData)[:,0:dataDimension],dtype=np.float64)
        #计算均值
        category[i].clusteringMeans = mean(dataArray,0)
        #更新聚类中心
        category[i].clusteringCenter = category[i].clusteringMeans
        #计算每一类内距离差
        error = calcu_error(dataArray,category[i].clusteringMeans)
        category[i].clusteringError = error
        J += error
    return J

def k_Means(dataPath,dataSize,dataDimension,K,testRate=0):
    #提取数据
    train_Data,test_Data,group1,group2,group3 = readData(dataPath,dataSize,dataDimension,testRate)
    print("train_Data SIZE:",len(train_Data))

    #随机选取K个聚类中心,并初始化聚类名称与聚类均值
    clustering_center_label = random.sample(range(0,len(train_Data)-1),K)
    print("clustering_center_label:",clustering_center_label)
    category = [Data_K_means() for i in range(K)]
    for i in range(K):
        category[i].clusteringCenter = train_Data[clustering_center_label[i]][:dataDimension]
        category[i].clusteringMeans = category[i].clusteringCenter
        category[i].name = "category_"+str(i)
        # category[i].print()

    J = 1024.0
    J_last = -1024.0
    repeat_time = 0
    #开始迭代
    while J!=J_last:    #以准则函数作为迭代结束依据
        J_last = J
        repeat_time += 1

        #按最小距离准则划分样本
        divide_Sample(train_Data,category,dataDimension,K)

        #计算均值，更新聚类中心，计算准则函数值
        J = update_Cluster_Center(category,dataDimension,K)

        print("---------------------------------------------------------------------------->Num: ",repeat_time)
        # for i in range(K):
        #     category[i].print()
        print("J = ",J)
        print("J_last = ",J_last,"\n\n")

    #迭代结束，统计分类结果标签数目 并 计算准确率
    accuracy = 0
    for k in range(K):
        _1_Num = _2_Num = 0.0
        for i in range(len(category[k].clusteringData)):
            if category[k].clusteringData[i][dataDimension] == 'R\n':
                _1_Num += 1
            if category[k].clusteringData[i][dataDimension] == 'M\n':
                _2_Num += 1
        category[k].print()
        print("R:",_1_Num)
        print("M:",_2_Num,"\n\n")

        category[k].R = _1_Num
        category[k].M = _2_Num

        if _1_Num>=_2_Num:
            accuracy += _1_Num/len(category[k].clusteringData) * len(category[k].clusteringData)/len(train_Data)
        else:
            accuracy += _2_Num/len(category[k].clusteringData) * len(category[k].clusteringData)/len(train_Data)


    return category,J,accuracy


def test_K_Means():
    J = [0]
    accuracy = [0]
    testTimes = 10

    #绘制聚类结果
    plt.figure()
    for i in range(testTimes):
        category,J_temp,accuracy_temp = k_Means("sonar.all-data",208,60,K=2)    #进行K均值聚类,返回聚类结果、误差平方和、精准度
        J.append(J_temp)
        accuracy.append(accuracy_temp)

        for k in range(2):
            p1 = plt.bar(x=k+(i*5), height=category[k].R, label='R', color="red")
            p2 = plt.bar(x=k+(i*5), height=category[k].M, label='M', color="green",bottom=category[k].R)
        plt.legend((p1[0], p2[0]), ('R', 'M'))
    plt.ylabel('Number')
    plt.xlabel("testTimes")
    plt.xticks([])
    plt.title('K-means result')
    plt.show(block = False)

    #绘制聚类精确度
    plt.figure()
    for i in range(1,testTimes+1):
        plt.scatter(i,accuracy[i])
        print("Accuracy_",i,": ",accuracy[i])
    print("Accuracy_AVE",mean(accuracy[1:testTimes]))
    plt.title("K-means Accuracy")
    plt.xlabel("testTimes")
    plt.ylabel("Accuracy")
    plt.plot(accuracy)
    plt.show(block = False)

    #绘制聚类最终准则函数值（#最小误差平方和）
    plt.figure()
    for i in range(1,testTimes+1):
        plt.scatter(i,J[i])
        print("J_",i,": ",J[i])
    plt.title("K-means J_e")
    plt.xlabel("testTimes")
    plt.ylabel("J_e")
    plt.plot(J)
    plt.show()

#************************************************************FCM********************************************************************
class Data_FCM:
    def __init__(self): 
        self.name = ''

        self.Data = []          #最终包含的样本数据
        self.Membership = []    #隶属度
        self.Means = 0          #均值/聚类中心

        self.R = 0
        self.M = 0
        self.Iris_virginica = 0

    def print(self):
        print("**************************   ",self.name,"   ***********************")

        print("clusteringMeans: ",self.Means)
        # print("clusteringData: \n"  ,np.array(self.clusteringData))
        print("dataSize: "  ,len(self.Data))

#计算 更新隶属度函数
def update_Membership(train_Data,category,dataDimension,C,b):
    trainDataArray = np.array(np.array(train_Data)[:,0:dataDimension],dtype=np.float64)
    for j in range(C):  #u_j(x_i)
        category[j].Membership = []     #清空隶属度向量
        for i in range(len(train_Data)):
            #计算分母、分子
            denom = 0.0
            for l in range(C):
                denom += pow((1/np.square(np.linalg.norm(np.array(trainDataArray[i])-np.array(category[l].Means)))),1/(b-1))
            numer = pow((1/np.square(np.linalg.norm(np.array(trainDataArray[i])-np.array(category[j].Means)))),1/(b-1))
            #更新隶属度
            if numer == inf:    #解决自己与自己隶属度计算出错问题
                category[j].Membership.append(1.0)
            else:
                category[j].Membership.append(numer/denom)
        # print(j,np.array(category[j].Membership))

#计算 更新聚类中心
def update_Cluster_Center_FCM(train_Data,category,dataDimension,C,b):
    trainDataArray = np.array(np.array(train_Data)[:,0:dataDimension],dtype=np.float64)
    for j in range(C):
        #计算分母、分子
        denom = 0.0
        numer = 0.0
        for i in range(len(train_Data)):
            denom += pow(category[j].Membership[i],b)
            numer += pow(category[j].Membership[i],b) * trainDataArray[i]
        # print(j,":",numer,denom,numer/denom)
        category[j].Means = numer/denom

#根据隶属度函数划分样本
def divide_Sample_FCM(train_Data,category,C):
    for i in range(len(train_Data)):
        membership_max = -1
        label = -1
        for j in range(C):
            if category[j].Membership[i] > membership_max:
                membership_max = category[j].Membership[i]
                label = j
        category[label].Data.append(train_Data[i])

def FCM(dataPath,dataSize,dataDimension,C,b=2,end_condi=0.001,testRate=0):
    #提取数据
    train_Data,test_Data,group1,group2,group3 = readData(dataPath,dataSize,dataDimension,testRate)
    print("train_Data SIZE:",len(train_Data))

    #随机选取C个聚类中心,初始化聚类中心(均值向量)
    clustering_center_label = random.sample(range(0,len(train_Data)-1),C)
    print("clustering_center_label:",clustering_center_label)
    category = [Data_FCM() for i in range(C)]
    for i in range(C):
        category[i].Means = train_Data[clustering_center_label[i]][:dataDimension]
        category[i].name = "category_"+str(i)
        category[i].print()

    m = [1024 for i in range(dataDimension)]
    m_last = [-1024 for i in range(dataDimension)]
    repeat_time = 0
    #开始迭代
    while calcu_Eucli_Dis(m,m_last) > end_condi:
        repeat_time += 1
        m_last = m

        #计算，更新隶属度
        update_Membership(train_Data,category,dataDimension,C,b)

        #计算均值，更新聚类中心
        update_Cluster_Center_FCM(train_Data,category,dataDimension,C,b)

        m = category[0].Means

        print("---------------------------------------------------------------------------->Num: ",repeat_time)
        print("m=",m)
        print("m_last = ",m_last)
    #迭代结束，划分样本
    divide_Sample_FCM(train_Data,category,C)

    #划分结束，统计分类结果标签数目 并 计算准确率
    accuracy = 0
    for k in range(C):
        _1_Num = _2_Num = 0
        for i in range(len(category[k].Data)):
            if category[k].Data[i][dataDimension] == 'R\n':
                _1_Num += 1
            if category[k].Data[i][dataDimension] == 'M\n':
                _2_Num += 1
        category[k].print()
        print("R:",_1_Num)
        print("M:",_2_Num,"\n\n")
        category[k].R = _1_Num
        category[k].M = _2_Num

        if _1_Num>=_2_Num:
            accuracy += _1_Num/len(category[k].Data) * len(category[k].Data)/len(train_Data)
        else:
            accuracy += _2_Num/len(category[k].Data) * len(category[k].Data)/len(train_Data)

    return category,accuracy

def test_FCM():
    accuracy = [0]
    testTimes = 10

    #绘制聚类结果
    plt.figure()
    for i in range(testTimes):
        category,accuracy_temp = FCM("sonar.all-data",208,60,C=2,b=2,end_condi=0.001)    #进行K均值聚类,返回聚类结果、误差平方和、精准度
        accuracy.append(accuracy_temp)

        for k in range(2):
            p1 = plt.bar(x=k+(i*5), height=category[k].R, label='R', color="red")
            p2 = plt.bar(x=k+(i*5), height=category[k].M, label='M', color="green",bottom=category[k].R)
        plt.legend((p1[0], p2[0]), ('R', 'M'))
    plt.ylabel('Number')
    plt.xlabel("testTimes")
    plt.xticks([])
    plt.title('FCM result')
    plt.show(block = False)

    #绘制聚类精确度
    plt.figure()
    for i in range(1,testTimes+1):
        plt.scatter(i,accuracy[i])
        print("Accuracy_",i,": ",accuracy[i])
    print("Accuracy_AVE",mean(accuracy[1:testTimes]))
    plt.title("FCM Accuracy")
    plt.xlabel("testTimes")
    plt.ylabel("Accuracy")
    plt.plot(accuracy)
    plt.show()


if __name__ == '__main__':
    # test_K_Means()
    test_FCM()

