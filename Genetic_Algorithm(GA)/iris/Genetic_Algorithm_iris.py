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
    _1_Data = np.mat(np.zeros((_1_Num,DATA_DIMENSION)))
    _2_Data = np.mat(np.zeros((_2_Num,DATA_DIMENSION)))
    _3_Data = np.mat(np.zeros((_3_Num,DATA_DIMENSION)))
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

#******************************************************* KNN ********************************************************
def dis_Sort(allDisInfo):
    for i in range(0,len(allDisInfo)): 
        for j in range(0,len(allDisInfo)-i-1):
            if allDisInfo[j][1]>allDisInfo[j+1][1]:
                temp = allDisInfo[j]
                allDisInfo[j] = allDisInfo[j+1]
                allDisInfo[j+1] = temp
    return allDisInfo

#计算欧氏距离
def calcu_Eucli_Dis(x,y):
    dis = np.sqrt(np.sum(np.square(np.array(x)-np.array(y))))
    return dis

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

def K_Nearest_Neighbors(train_Data,test_Data,dataDimension,k=5):
    #数据准备
    # print("*********************************************************************")
    # print("                    train_Size:",len(train_Data))
    # print("                    test_Size:",len(test_Data))
    # print("*********************************************************************")
    
    wrongNum = correctNum = 0
    #遍历每一个测试元素
    for testElement in range(len(test_Data)):
        allDisInfo = [[0 for i in range(2)] for j in range(len(train_Data))]  #[forcast_label][distance]

        # print("**********************testElement_",testElement,"************************")
        #每一个测试元素与训练集中的每一个数据进行距离计算
        for trainElement in range(len(train_Data)):
            testArray = np.array(test_Data[testElement][0:dataDimension],float)
            trainArray = np.array(train_Data[trainElement][0:dataDimension],float)
            #计算距离
            dis = calcu_Eucli_Dis(testArray,trainArray)
            #保存所有label的距离信息
            allDisInfo[trainElement][0] = train_Data[trainElement][dataDimension]
            allDisInfo[trainElement][1] = dis
        
        #对距离信息进行排序
        dis_Sort(allDisInfo)
        # for i in range(len(allDisInfo)):
        #     print(allDisInfo[i])
        #选择K值进行判别
        forcast_label = k_Judge(allDisInfo,k)
        # print("------------------------------------> Forcast Label:",forcast_label,"-----------------------------------> Real Label:",test_Data[testElement][dataDimension])
        if(test_Data[testElement][dataDimension] == forcast_label):
            correctNum += 1
        else:
            wrongNum += 1

    # print("------------------------------------")
    # print("|WrongNums:",wrongNum,"correctNums:",correctNum)
    # print("------------------------------------\n\n\n")
    Accuracy = correctNum/(correctNum+wrongNum)
    return  Accuracy

#************************************************************* Genetic Algorithm ********************************************************************
# 每个个体的类
class individual:
    def __init__(self,dataDimension,choiceDimension):
        super().__init__()
        init_chrom_val = random.sample(range(0,dataDimension),choiceDimension)
        print(init_chrom_val)
        self.dataDimension = dataDimension  # 原始数据维度
        self.fitness = 0                    # 适应度
        self.trainArray = []
        self.testArray = []
        self.Chromosome = [0 for i in range(dataDimension)]        # 初始化染色体,随机选中d个特征
        for i in range(choiceDimension):
            self.Chromosome[init_chrom_val[i]] = 1
    
    def caclu_fitness(self,trainData,testData):
        deletePos = []
        for i in range(len(self.Chromosome)):
            if not self.Chromosome[i]:
                deletePos.append(i)

        self.trainArray = np.delete(np.array(trainData),deletePos,1).tolist()
        self.testArray = np.delete(np.array(testData),deletePos,1).tolist()

        self.fitness = K_Nearest_Neighbors(self.trainArray,self.testArray,self.dataDimension-len(deletePos))

def seletction(population): # 使用轮盘赌进行个体选择
    next_population = []

    # 归一化所有个体的适应度函数，以进行轮盘赌选择
    fitness_sum = 0
    fitness_norm = [0 for i in range(len(population))]
    for i in range(len(population)):
        fitness_sum += population[i].fitness

    start_pos = 0
    for i in range(len(population)):
        fitness_norm[i] = population[i].fitness/fitness_sum + start_pos
        start_pos = fitness_norm[i]

    for indiv in range(len(population)):    # 选出新一代种群，种群规模保持不变
        rand_num = random.random()
        for j in range(len(population)):
            if rand_num <= fitness_norm[j]:
                break

        print("individual_",j,"_:is selected")
        next_population.append(population[j])
    return next_population


# 交叉
def cross(population,cross_prob = 0.6):
    for i in range(len(population)):
        if random.random() <= cross_prob:
            # 一次重组D//2个基因，并且要保证交叉的染色体序列长度相同，含选中特征的数量相同

            # 统计交叉染色体中1的数量
            gen_1_num = 0
            for gen in range(len(population[i].Chromosome)//2):
                if population[i].Chromosome[gen] == 1:
                    gen_1_num += 1

            # 随机寻找另一个个体，并寻找到相匹配的染色体片段
            another_indiv = random.randint(0,len(population)-1) # 与之交配的个体
            another_cross_pos = 0
            while 1:
                # print(i,another_indiv,another_cross_pos)
                another_gen_1_num = 0
                for gen in range(another_cross_pos,len(population[i].Chromosome)//2+another_cross_pos):
                    if population[another_indiv].Chromosome[gen] == 1:
                        another_gen_1_num += 1

                if another_gen_1_num == gen_1_num:
                    print(i,another_indiv,"pos",another_cross_pos)
                    tmp = population[i].Chromosome[0:len(population[i].Chromosome)//2]
                    population[i].Chromosome[0:len(population[i].Chromosome)//2] = population[another_indiv].Chromosome[another_cross_pos:len(population[i].Chromosome)//2+another_cross_pos]
                    population[another_indiv].Chromosome[another_cross_pos:len(population[i].Chromosome)//2+another_cross_pos] = tmp
                    break
                else:
                    another_cross_pos += 1

                if len(population[i].Chromosome)//2+another_cross_pos > len(population[i].Chromosome):  #超出范围，寻找失败
                    print(i,"fail to find another position")
                    break
# 存在自交问题
def cross_v2(population,cross_prob = 0.4):
    for i in range(len(population)//2):
        if random.random() <= cross_prob:
            # 一次重组D//2个基因，并且要保证交叉的染色体序列长度相同，含选中特征的数量相同

            # 统计交叉染色体中1的数量
            cross_pos = random.randint(0,len(population[i].Chromosome)//2-1)
            cross_pos_stop = cross_pos+len(population[i].Chromosome)//2
            gen_1_num = 0
            for gen in range(cross_pos,cross_pos_stop):
                if population[i].Chromosome[gen] == 1:
                    gen_1_num += 1

            # 随机寻找另一个个体，并寻找到相匹配的染色体片段
            another_indiv = i
            while i == another_indiv:
                another_indiv = random.randint(0,len(population)//2-1) # 与之交配的个体

            another_cross_pos = 0
            another_cross_pos_stop = len(population[i].Chromosome)//2+another_cross_pos
            # print("stop",another_cross_pos_stop)
            while 1:
                another_gen_1_num = 0
                for gen in range(another_cross_pos,another_cross_pos_stop):
                    if population[another_indiv].Chromosome[gen] == 1:
                        another_gen_1_num += 1
                if another_gen_1_num == gen_1_num:
                    print("individual_",i,"_and individual_",another_indiv,"_ have crossed at positon: 0 and position:",another_cross_pos)
                    # print("yuan:",population[i].Chromosome[cross_pos:cross_pos_stop],len(population[i].Chromosome[cross_pos:cross_pos_stop]))
                    # print("huan:",population[another_indiv].Chromosome[another_cross_pos:another_cross_pos_stop],len(population[another_indiv].Chromosome[another_cross_pos:another_cross_pos_stop]))
                    tmp = population[i].Chromosome[cross_pos:cross_pos_stop]
                    population[i].Chromosome[cross_pos:cross_pos_stop] = population[another_indiv].Chromosome[another_cross_pos:another_cross_pos_stop]
                    population[another_indiv].Chromosome[another_cross_pos:another_cross_pos_stop] = tmp
                    break
                else:
                    another_cross_pos += 1
                    another_cross_pos_stop += 1

                if another_cross_pos_stop >= len(population[i].Chromosome):  #超出范围，寻找失败
                    print(i,"fail to find another position")
                    break
# 变异
def variation(population,variation_prob = 0.01):
    for i in range(len(population)):
        if random.random() <= variation_prob:
            variation_pos = random.randint(0,len(population[i].Chromosome)-1)      # 变异的位置
            another_pos = random.randint(0,len(population[i].Chromosome)-1)
            # 变异后要保证选中的特征数量不变
            if (population[i].Chromosome[variation_pos] == 1):  #变异位置原来为1就找到位置为0的也变异
                while population[i].Chromosome[another_pos] != 0:
                    another_pos = random.randint(0,len(population[i].Chromosome)-1)
                population[i].Chromosome[another_pos] = 1
                population[i].Chromosome[variation_pos] = 0
            else:                                               #变异位置原来为0就找到位置为1的也变异
                while population[i].Chromosome[another_pos] != 1:
                    another_pos = random.randint(0,len(population[i].Chromosome)-1)
                population[i].Chromosome[another_pos] = 0
                population[i].Chromosome[variation_pos] = 1
            print("individual_",i,"_: have variated,position:[",variation_pos,another_pos,"]")


def GA(dataPath,dataSize,dataDimension,choiceDimension,populationSize,STOP_TIMES):
    # 加载数据集
    trainData,testData,_,_,_ = readData(dataPath,dataSize,dataDimension,1/5)
    # 初始化种群 和 最优个体
    population = [individual(dataDimension,choiceDimension) for i in range(populationSize)]

    fit_ave = [0]
    interation_times = 0    #迭代次数
    # 开始迭代：
    while interation_times < STOP_TIMES:
        great_indiv = population[0]
        fitness_ave = 0
        interation_times += 1
        print("************************ _generation:",interation_times,"_***************************")
        # 更新每个个体的适应度函数,并更新最优个体
        for i in range(populationSize): 
            population[i].caclu_fitness(trainData,testData)
            fitness_ave += population[i].fitness

            if population[i].fitness > great_indiv.fitness:
                great_indiv = population[i]
            print("individual_",i,"_:",population[i].Chromosome,population[i].fitness)
        print("best:",great_indiv.Chromosome,great_indiv.fitness)

        fitness_ave /= populationSize
        fit_ave.append(fitness_ave)

        # 轮盘赌选择
        population = seletction(population)

        # 交叉
        cross(population, cross_prob = 0.4)

        # 变异
        variation(population, variation_prob = 0.01)

    return great_indiv,fit_ave

# 同一维度看收敛情况
if __name__ == "__main__":
    great_indiv,fit_ave = GA("iris.data",150,4,choiceDimension=2,populationSize=10,STOP_TIMES=20)
    for i in range(len(fit_ave)):
        print("Fitness",i,": ",fit_ave[i])
        plt.scatter(i,fit_ave[i])
    plt.title("Changes of population fitness with the number of iterations")
    plt.xlabel("Iteration times")
    plt.ylabel("Fitness")
    # plt.ylim(0,2)
    plt.plot(fit_ave)
    plt.show()

# 不同维度
# if __name__ == '__main__':
#     Accuracy_AVE = 0
#     Accuracy = []
#     Accuracy.append(0)
#     for dimen in range(1,5):
#         great_indiv,_ = GA("iris.data",150,4,choiceDimension=dimen,populationSize=5,STOP_TIMES=10)
#         Accuracy.append(great_indiv.fitness)
#         plt.scatter(dimen,great_indiv.fitness)
#         Accuracy_AVE += great_indiv.fitness
#     Accuracy_AVE /= 4

#     for i in range(5):
#         print("Accuracy_",i+1,": ",Accuracy[i])
#     print("Accuracy_AVE:",Accuracy_AVE)
#     plt.title("Accuracy")
#     plt.xlabel("choiceDimension")
#     plt.ylabel("Fitness")
#     # plt.ylim(0,2)
#     plt.plot(Accuracy)

    # plt.show()