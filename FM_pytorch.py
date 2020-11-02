#_*_coding:utf-8_*_
'''
@project: shuidi
@author: exudingtao
@time: 2020/10/21 2:43 下午
'''


import numpy as np
from random import normalvariate #正态分布
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


# 此函数在此数据集下没用上，因特征不需要改动
def process_feat(data, dense_feats, sparse_feats):
    '''
    :param data: df格式
    :param dense_feats: dense特征取对数
    :param sparse_feats: sparse特征进行类别编码
    :return:
    '''
    df = data.copy()
    # dense
    df_dense = df[dense_feats].fillna(0.0)
    for f in tqdm(dense_feats):
        df_dense[f] = df_dense[f].apply(lambda x: np.log(1 + x) if x > -1 else -1)

    # sparse
    df_sparse = df[sparse_feats].fillna('-1')
    for f in tqdm(sparse_feats):
        lbe = LabelEncoder()
        df_sparse[f] = lbe.fit_transform(df_sparse[f])

    df_new = pd.concat([df_dense, df_sparse], axis=1)
    return df_new


def preprocessing_min_max(x_train,x_test):
    scaler = MinMaxScaler()
    sca = scaler.fit(x_train)
    x_train_transform = sca.transform(x_train)
    x_test_transform = sca.transform(x_test)
    return x_train_transform,x_test_transform


def preprocessing(data_input):
    standardopt = MinMaxScaler()
    data_input.iloc[:, -1].replace(0, -1, inplace=True) #把数据集中的0转为-1
    feature = data_input.iloc[:, :-1] #除了最后一列之外，其余均为特征
    feature = standardopt.fit_transform(feature) #将特征转换为0与1之间的数
    feature = np.mat(feature)#传回来的是array，如果要dataframe那用dataframe
    label = np.array(data_input.iloc[:, -1]) #最后一列是标签，表示有无糖尿病
    return feature, label #返回特征，标签


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def sgd_fm(datamatrix, label, k, iter, alpha):
    '''
    k：分解矩阵的长度
    datamatrix：数据集特征
    label：数据集标签
    iter:迭代次数
    alpha:学习率
    '''
    m, n = np.shape(datamatrix) #m:数据集特征的行数，n:数据集特征的列数
    w0 = 0.0 #初始化w0为0
    w = np.zeros((n, 1)) #初始化w
    v = normalvariate(0, 0.2) * np.ones((n, k))
    for it in range(iter):
        for i in range(m):
            # inner1 = datamatrix[i] * w
            inner1 = datamatrix[i] * v #对应公式进行计算
            inner2 = np.multiply(datamatrix[i], datamatrix[i]) * np.multiply(v, v)
            jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
            ypredict = w0 + datamatrix[i] * w + jiaocha
            # print(np.shape(ypredict))
            # print(ypredict[0, 0])
            yp = sigmoid(label[i]*ypredict[0, 0])
            loss = 1 - (-(np.log(yp)))
            w0 = w0 - alpha * (yp - 1) * label[i] * 1
            for j in range(n):
                if datamatrix[i, j] != 0:
                    w[j] = w[j] - alpha * (yp - 1) * label[i] * datamatrix[i, j]
                    for k in range(k):
                        v[j, k] = v[j, k] - alpha * ((yp - 1) * label[i] * \
                                  (datamatrix[i, j] * inner1[0, k] - v[j, k] * \
                                  datamatrix[i, j] * datamatrix[i, j]))
        print('第%s次训练的误差为：%f' % (it, loss))
    return w0, w, v


def predict(w0, w, v, x, thold):
    inner1 = x * v
    inner2 = np.multiply(x, x) * np.multiply(v, v)
    jiaocha = np.sum((np.multiply(inner1, inner1) - inner2), axis=1) / 2.0
    ypredict = w0 + x * w + jiaocha
    y0 = sigmoid(ypredict[0,0])
    if y0 > thold:
        yp = 1
    else:
        yp = -1
    return yp


def calaccuracy(datamatrix, label, w0, w, v, thold):
    error = 0
    for i in range(np.shape(datamatrix)[0]):
        yp = predict(w0, w, v, datamatrix[i], thold)
        if yp != label[i]:
            error += 1
    accuray = 1.0 - error/np.shape(datamatrix)[0]
    return accuray



# 读取数据
print('loading data...')
#训练集500行，测试集268行。数据集共9列，前8列为数值特征值，最后一列为label[0,1]
data_train = pd.read_csv('./data/diabetes_train.txt', header=None)
data_test = pd.read_csv('./data/diabetes_test.txt', header=None)

print(data_train.shape, data_test.shape)
print(data_train.head())

x_train, y_train = preprocessing(data_train) #将训练集进行预处理，datamattrain存放训练集特征，labeltrain存放训练集标签
x_test, y_test = preprocessing(data_test)#将测试集进行预处理，datamattest存放训练集特征，labeltest存放训练集标签

w0, w, v = sgd_fm(x_train, y_train, 20, 300, 0.01)#分解矩阵的长度为20，迭代次数为300次，学习率为0.01
maxaccuracy = 0.0
tmpthold = 0.0
for i in np.linspace(0.4, 0.6, 201):
    accuracy_test = calaccuracy(x_test, y_test, w0, w, v, i)
    if accuracy_test > maxaccuracy:
        maxaccuracy = accuracy_test
        tmpthold = i
print("准确率:", accuracy_test)
