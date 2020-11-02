#_*_coding:utf-8_*_
'''
@project: shuidi
@author: exudingtao
@time: 2020/10/27 11:47 上午
'''

import pandas as pd
import numpy as np
import warnings
import random, math, os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#import faiss
import matplotlib.pyplot as plt
import torch
import torch.utils.data as Data



warnings.filterwarnings('ignore')


# 评价指标
# 推荐系统推荐正确的商品数量占用户实际点击的商品数量
def Recall(Rec_dict, Val_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rel_set)

    return round(hit_items / all_items * 100, 2)


# 推荐系统推荐正确的商品数量占给用户实际推荐的商品数
def Precision(Rec_dict, Val_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    Val_dict: 用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    hit_items = 0
    all_items = 0
    for uid, items in Val_dict.items():
        rel_set = items
        rec_set = Rec_dict[uid]
        for item in rec_set:
            if item in rel_set:
                hit_items += 1
        all_items += len(rec_set)

    return round(hit_items / all_items * 100, 2)


# 所有被推荐的用户中,推荐的商品数量占这些用户实际被点击的商品数量
def Coverage(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    rec_items = set()
    all_items = set()
    for uid in Rec_dict:
        for item in Trn_dict[uid]:
            all_items.add(item)
        for item in Rec_dict[uid]:
            rec_items.add(item)
    return round(len(rec_items) / len(all_items) * 100, 2)


# 使用平均流行度度量新颖度,如果平均流行度很高(即推荐的商品比较热门),说明推荐的新颖度比较低
def Popularity(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: {item1, item2,...}, uid: {item1, item2,...}, ...}
    '''
    pop_items = {}
    for uid in Trn_dict:
        for item in Trn_dict[uid]:
            if item not in pop_items:
                pop_items[item] = 0
            pop_items[item] += 1

    pop, num = 0, 0
    for uid in Rec_dict:
        for item in Rec_dict[uid]:
            pop += math.log(pop_items[item] + 1)  # 物品流行度分布满足长尾分布,取对数可以使得平均值更稳定
            num += 1
    return round(pop / num, 3)


# 将几个评价指标指标函数一起调用
def rec_eval(val_rec_items, val_user_items, trn_user_items):
    print('recall:', Recall(val_rec_items, val_user_items))
    print('precision', Precision(val_rec_items, val_user_items))
    print('coverage', Coverage(val_rec_items, trn_user_items))
    print('Popularity', Popularity(val_rec_items, trn_user_items))


def get_data(root_path):
    # 读取数据时，定义的列名
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(os.path.join(root_path, 'ratings.dat'), sep='::', engine='python', names=rnames)

    lbe = LabelEncoder()
    data['user_id'] = lbe.fit_transform(data['user_id'])
    data['movie_id'] = lbe.fit_transform(data['movie_id'])

    # 直接这么分是不是可能会存在验证集中的用户或者商品不在训练集合中呢？那这种的操作一半是怎么进行划分
    trn_data_, val_data_, _, _ = train_test_split(data, data, test_size=0.2)

    trn_data = trn_data_.groupby('user_id')['movie_id'].apply(list).reset_index()
    val_data = val_data_.groupby('user_id')['movie_id'].apply(list).reset_index()

    trn_user_items = {}
    val_user_items = {}

    # 将数组构造成字典的形式{user_id: [item_id1, item_id2,...,item_idn]}
    for user, movies in zip(*(list(trn_data['user_id']), list(trn_data['movie_id']))):
        trn_user_items[user] = set(movies)

    for user, movies in zip(*(list(val_data['user_id']), list(val_data['movie_id']))):
        val_user_items[user] = set(movies)

    return trn_user_items, val_user_items, trn_data_, val_data_, data


def normalizeRating(rating_train):
    m,n=rating_train.shape
    # 每部电影的平均得分
    rating_mean=torch.zeros((m,1))
    #所有电影的评分
    all_mean=0
    for i in range(m):
        #每部电影的评分
        idx=(rating_train[i,:]!=0)
        rating_mean[i]=torch.mean(rating_train[i,idx])
    tmp=rating_mean.numpy()
    tmp=np.nan_to_num(tmp)        #对值为NaN进行处理，改成数值0
    rating_mean=torch.tensor(tmp)
    no_zero_rating=np.nonzero(tmp)                #numpyy提取非0元素的位置
    # print("no_zero_rating:",no_zero_rating)
    no_zero_num=np.shape(no_zero_rating)[1]   #非零元素的个数
    print("no_zero_num:",no_zero_num)
    all_mean=torch.sum(rating_mean)/no_zero_num
    return rating_mean,all_mean


# 矩阵分解模型
class MF(torch.nn.Module):
    def __init__(self, userNo, itemNo, num_feature=64):
        super(MF, self).__init__()
        self.num_feature = num_feature     #num of laten features
        self.userNo = userNo               #user num
        self.itemNo = itemNo               #item num
        self.bi = torch.nn.Parameter(torch.rand(self.itemNo, 1))    #parameter
        self.bu = torch.nn.Parameter(torch.rand(self.userNo, 1))    #parameter
        self.U = torch.nn.Parameter(torch.rand(self.num_feature, self.userNo))    #parameter
        self.V = torch.nn.Parameter(torch.rand(self.itemNo, self.num_feature))    #parameter

    def mf_layer(self,train_set=None):
        # predicts=all_mean+self.bi+self.bu.t()+pt.mm(self.V,self.U)
        predicts = self.bi + self.bu.t() + torch.mm(self.V, self.U)
        return predicts

    def forward(self, train_set):
        output = self.mf_layer(train_set)
        return output


if __name__ == "__main__":
    # K表示最终给用户推荐的商品数量，N表示候选推荐商品为用户交互过的商品相似商品的数量
    k = 80
    N = 10
    BATCH_SIZE = 100

    # 读取数据
    root_path = './data/ml-1m/'
    trn_user_items, val_user_items, trn_data, val_data, data = get_data(root_path)

    # 去掉时间戳
    train = trn_data.drop(['timestamp'], axis=1)
    test = val_data.drop(['timestamp'], axis=1)
    print("train shape:", train.shape)
    print("test shape:", test.shape)

    # userNo的最大值
    userNo = max(train['user_id'].max(), test['user_id'].max()) + 1
    print("userNo:", userNo)
    # movieNo的最大值
    itemNo = max(train['movie_id'].max(), test['movie_id'].max()) + 1
    print("itemNo:", itemNo)

    rating_train = torch.zeros((itemNo, userNo))
    rating_test = torch.zeros((itemNo, userNo))
    for index, row in train.iterrows():
        # train数据集进行遍历
        rating_train[int(row['movie_id'])][int(row['user_id'])] = row['rating']
    print(rating_train[0:3][1:10])
    for index, row in test.iterrows():
        rating_test[int(row['movie_id'])][int(row['user_id'])] = row['rating']


    def normalizeRating(rating_train):
        m, n = rating_train.shape
        # 每部电影的平均得分
        rating_mean = torch.zeros((m, 1))
        # 所有电影的评分
        all_mean = 0
        for i in range(m):
            # 每部电影的评分
            idx = (rating_train[i, :] != 0)
            rating_mean[i] = torch.mean(rating_train[i, idx])
        tmp = rating_mean.numpy()
        tmp = np.nan_to_num(tmp)  # 对值为NaN进行处理，改成数值0
        rating_mean = torch.tensor(tmp)
        no_zero_rating = np.nonzero(tmp)  # numpyy提取非0元素的位置
        # print("no_zero_rating:",no_zero_rating)
        no_zero_num = np.shape(no_zero_rating)[1]  # 非零元素的个数
        print("no_zero_num:", no_zero_num)
        all_mean = torch.sum(rating_mean) / no_zero_num
        return rating_mean, all_mean


    rating_mean, all_mean = normalizeRating(rating_train)
    print("all mean:", all_mean)

    # 训练集分批处理
    loader = Data.DataLoader(
        dataset=rating_train,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # 最新批数据
        shuffle=False  # 是否随机打乱数据
    )

    loader2 = Data.DataLoader(
        dataset=rating_test,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # 最新批数据
        shuffle=False  # 是否随机打乱数据
    )

    num_feature = 2  # k
    mf = MF(userNo, itemNo, num_feature)
    print("parameters len:", len(list(mf.parameters())))
    param_name = []
    params = []
    for name, param in mf.named_parameters():
        param_name.append(name)
        print(name)
        params.append(param)
    # param_name的参数依次为bi,bu,U,V

    lr = 0.3
    _lambda = 0.001
    loss_list = []
    optimizer = torch.optim.SGD(mf.parameters(), lr)
    # 对数据集进行训练
    for epoch in range(1000):
        optimizer.zero_grad()
        output = mf(train)
        loss_func = torch.nn.MSELoss()
        # loss=loss_func(output,rating_train)+_lambda*(pt.sum(pt.pow(params[2],2))+pt.sum(pt.pow(params[3],2)))
        loss = loss_func(output, rating_train)
        loss.backward()
        optimizer.step()
        loss_list.append(loss)

    print("train loss:", loss)


    # 评价指标rmse
    def rmse(pred_rate, real_rate):
        # 使用均方根误差作为评价指标
        loss_func = torch.nn.MSELoss()
        mse_loss = loss_func(pred_rate, real_rate)
        rmse_loss = torch.sqrt(mse_loss)
        return rmse_loss


    # 测试网络
    # 测试时测试的是原来评分矩阵为0的元素，通过模型将为0的元素预测一个评分，所以需要找寻评分矩阵中原来元素为0的位置。
    prediction = output[np.where(rating_train == 0)]
    # 评分矩阵中元素为0的位置对应测试集中的评分
    rating_test = rating_test[np.where(rating_train == 0)]
    rmse_loss = rmse(prediction, rating_test)
    print("test loss:", rmse_loss)

    plt.clf()
    plt.plot(range(epoch + 1), loss_list, label='Training data')
    plt.title("The MovieLens Dataset Learning Curve")
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()