from scipy.io import loadmat
from svmutil import *
import random

# 读取数据，并随机分为测试集和训练集
def process_dataset(test_rate):

    m = loadmat("DataSet/YaleB_32x32.mat")
    feature = m['fea']
    label = m['gnd']
    print(m.keys())

    test_Labels = random.sample(range(0,2414-1),int(2414*test_rate))

    f_train = open("DataSet/train_data.txt",'w')
    f_test = open("DataSet/test_data.txt",'w')
    for i in range(len(feature)):
        if i in test_Labels:
            f = f_test
        else:
            f = f_train

        f.write(str(label[i][0]) + ' ')
        for j in range(len(feature[i])):
            f.write("%d:%d " %(j+1,int(feature[i][j])))
        f.write('\n')

    # 输出所有数据
    # f = open("DataSet/YaleB.data",'w')
    # for i in range(len(feature)):
    #     f.write(str(label[i][0]) + ' ')
    #     for j in range(len(feature[i])):
    #         f.write("%5d:%d" %(j+1,int(feature[i][j])))
    #     f.write('\n')
    # f.close()

if __name__ == "__main__":
    # process_dataset(1/5)

    y_train,x_train = svm_read_problem("DataSet/train_data.txt")
    y_test,x_test = svm_read_problem("DataSet/test_data.txt")
    model = svm_train(y_train[:],x_train[:],)
    # model = svm_load_model("SVM_trained_model")

    svm_save_model("SVM_trained_model",model)
    
    p_label, p_acc, p_val = svm_predict(y_test[:], x_test[:], model)
    print(p_label)