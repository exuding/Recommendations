#_*_coding:utf-8_*_
'''
@project: 
@author: exudingtao
@time: 2020/10/21 3:40 下午
'''
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time as time
import click
import random
import collections


class CriteoDataset(Dataset):
    """
    Custom dataset class for Criteo dataset in order to use efficient
    dataloader tool provided by PyTorch.
    """

    def __init__(self, root, train=True):
        """
        Initialize file path and train/test mode.
        Inputs:
        - root: Path where the processed data file stored.
        - train: Train or test. Required.
        """
        self.root = root
        self.train = train

        if not self._check_exists:
            raise RuntimeError('Dataset not found.')

        if self.train:
            data = pd.read_csv(os.path.join(root, 'train.txt'))
            self.train_data = data.iloc[:, :-1].values
            self.target = data.iloc[:, -1].values
        else:
            data = pd.read_csv(os.path.join(root, 'test.txt'))
            self.test_data = data.iloc[:, :-1].values

    def __getitem__(self, idx):
        if self.train:
            dataI, targetI = self.train_data[idx, :], self.target[idx]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)  # 最后一列增加一个维度
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv, targetI
        else:
            dataI = self.test_data.iloc[idx, :]
            Xi = torch.from_numpy(dataI.astype(np.int32)).unsqueeze(-1)
            Xv = torch.from_numpy(np.ones_like(dataI))
            return Xi, Xv

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):  # 私有方法
        return os.path.exists(self.root)


class DeepFM(nn.Module):
    """
    A DeepFM network with RMSE loss for rates prediction problem.

    There are two parts in the architecture of this network: fm part for low
    order interactions of features and deep part for higher order. In this
    network, we use bachnorm and dropout technology for all hidden layers,
    and "Adam" method for optimazation.

    You may find more details in this paper:
    DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
    """

    def __init__(self, feature_sizes, embedding_size=4,
                 hidden_dims=[32, 32], num_classes=10, dropout=[0.5, 0.5],
                 use_cuda=True, verbose=False):
        """
        Initialize a new network

        Inputs:
        - feature_size: A list of integer giving the size of features for each field.
        - embedding_size: An integer giving size of feature embedding.
        - hidden_dims: A list of integer giving the size of each hidden layer.
        - num_classes: An integer giving the number of classes to predict. For example,
                    someone may rate 1,2,3,4 or 5 stars to a film.
        - batch_size: An integer giving size of instances used in each interation.
        - use_cuda: Bool, Using cuda or not
        - verbose: Bool
        """
        super().__init__()
        self.field_size = len(feature_sizes)
        self.feature_sizes = feature_sizes
        self.embedding_size = embedding_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dtype = torch.float

        """
            check if use cuda
        """
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        """
            init fm part
        """
        """
        这里主要提醒的就是数据格式问题，因为我们的前13行数据是连续型数据，是float型的。
        这种数据想进入embedding layer需要做离散化处理，
        这里我延续了它连续性变量的本质，前13列数据使用的是dense layer的概念（也就是全连接神经网络层），
        这个层在keras建立使用的是 Dense()，在pytorch是 nn.Linear()
        所以步骤大的方向是将数据分列（这里是39列）输入对应的层，最后再来做合并。

        """

        #        self.fm_first_order_embeddings = nn.ModuleList(
        #            [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
        fm_first_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])
        fm_first_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])
        self.fm_first_order_models = fm_first_order_Linears.extend(fm_first_order_embeddings)

        #        self.fm_second_order_embeddings = nn.ModuleList(
        #            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
        fm_second_order_Linears = nn.ModuleList(
            [nn.Linear(feature_size, self.embedding_size) for feature_size in self.feature_sizes[:13]])
        fm_second_order_embeddings = nn.ModuleList(
            [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes[13:40]])
        self.fm_second_order_models = fm_second_order_Linears.extend(fm_second_order_embeddings)

        """
            init deep part
        """
        all_dims = [self.field_size * self.embedding_size] + \
                   self.hidden_dims + [self.num_classes]
        for i in range(1, len(hidden_dims) + 1):
            setattr(self, 'linear_' + str(i),
                    nn.Linear(all_dims[i - 1], all_dims[i]))
            # nn.init.kaiming_normal_(self.fc1.weight)
            setattr(self, 'batchNorm_' + str(i),
                    nn.BatchNorm1d(all_dims[i]))
            setattr(self, 'dropout_' + str(i),
                    nn.Dropout(dropout[i - 1]))

    def forward(self, Xi, Xv):
        """
        Forward process of network.

        Inputs:
        - Xi: A tensor of input's index, shape of (N, field_size, 1)
        - Xv: A tensor of input's value, shape of (N, field_size, 1)
        """
        """
            fm part
        """
        emb = self.fm_first_order_models[20]
        #        print(Xi.size())
        for num in Xi[:, 20, :][0]:
            if num > self.feature_sizes[20]:
                print("index out")

        #        fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_first_order_models)]
        #        fm_first_order_emb_arr = [(emb(Xi[:, i, :]) * Xv[:, i])  for i, emb in enumerate(self.fm_first_order_models)]
        fm_first_order_emb_arr = []
        for i, emb in enumerate(self.fm_first_order_models):
            if i <= 12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_first_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())
        #        print("successful")
        #        print(len(fm_first_order_emb_arr))
        fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        #        fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in enumerate(self.fm_second_order_models)]
        # fm_second_order_emb_arr = [(emb(Xi[:, i]) * Xv[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_emb_arr = []
        for i, emb in enumerate(self.fm_second_order_models):
            if i <= 12:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.float)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem).unsqueeze(1), 1).t() * Xv[:, i]).t())
            else:
                Xi_tem = Xi[:, i, :].to(device=self.device, dtype=torch.long)
                fm_second_order_emb_arr.append((torch.sum(emb(Xi_tem), 1).t() * Xv[:, i]).t())

        fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        fm_sum_second_order_emb_square = fm_sum_second_order_emb * \
                                         fm_sum_second_order_emb  # (x+y)^2
        fm_second_order_emb_square = [
            item * item for item in fm_second_order_emb_arr]
        fm_second_order_emb_square_sum = sum(
            fm_second_order_emb_square)  # x^2+y^2
        fm_second_order = (fm_sum_second_order_emb_square -
                           fm_second_order_emb_square_sum) * 0.5
        """
            deep part Deep部分
            其实就是两个带着BatchNorm和Dropout的全连接层，setattr就是让这两个层跟在了前面的模型后边。
        """
        #        print(len(fm_second_order_emb_arr))
        #        print(torch.cat(fm_second_order_emb_arr, 1).shape)
        deep_emb = torch.cat(fm_second_order_emb_arr, 1)
        deep_out = deep_emb
        for i in range(1, len(self.hidden_dims) + 1):
            deep_out = getattr(self, 'linear_' + str(i))(deep_out)
            deep_out = getattr(self, 'batchNorm_' + str(i))(deep_out)
            deep_out = getattr(self, 'dropout_' + str(i))(deep_out)
        #            print("successful")
        """
            sum 整合部分，将前面的几大部分特征作为一个整合来作为模型的输出，方便后面跟 label 比对进行学习。
        """
        #        print("1",torch.sum(fm_first_order, 1).shape)
        #        print("2",torch.sum(fm_second_order, 1).shape)
        #        print("deep",torch.sum(deep_out, 1).shape)
        #        print("bias",bias.shape)
        bias = torch.nn.Parameter(torch.randn(Xi.size(0)))
        total_sum = torch.sum(fm_first_order, 1) + \
                    torch.sum(fm_second_order, 1) + \
                    torch.sum(deep_out, 1) + bias
        return total_sum

    def fit(self, loader_train, loader_val, optimizer, epochs=1, verbose=False, print_every=5):
        """
        Training a model and valid accuracy.

        Inputs:
        - loader_train: I
        - loader_val: .
        - optimizer: Abstraction of optimizer used in training process, e.g., "torch.optim.Adam()""torch.optim.SGD()".
        - epochs: Integer, number of epochs.
        - verbose: Bool, if print.
        - print_every: Integer, print after every number of iterations.
        """
        """
            load input data
        """
        model = self.train().to(device=self.device)
        criterion = F.binary_cross_entropy_with_logits

        for epoch in range(epochs):
            for t, (xi, xv, y) in enumerate(loader_train):
                xi = xi.to(device=self.device, dtype=self.dtype)
                xv = xv.to(device=self.device, dtype=torch.float)
                y = y.to(device=self.device, dtype=self.dtype)

                total = model(xi, xv)
                #                print(total.shape)
                #                print(y.shape)
                loss = criterion(total, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if verbose and t % print_every == 0:
                    print('Epoch %d Iteration %d, loss = %.4f' % (epoch, t, loss.item()))
                    self.check_accuracy(loader_val, model)
                    print()

    def check_accuracy(self, loader, model):
        if loader.dataset.train:
            print('Checking accuracy on validation set')
        else:
            print('Checking accuracy on test set')
        num_correct = 0
        num_samples = 0
        model.eval()  # set model to evaluation mode
        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(device=self.device, dtype=self.dtype)  # move to device, e.g. GPU
                xv = xv.to(device=self.device, dtype=self.dtype)
                y = y.to(device=self.device, dtype=self.dtype)
                total = model(xi, xv)
                preds = (F.sigmoid(total) > 0.5).to(dtype=self.dtype)
                #                print(preds.dtype)
                #                print(y.dtype)
                #                print(preds.eq(y).cpu().sum())
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
            #                print("successful")
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))


# 将原始数据（raw data）处理成模型的输入数据
"""
主要在做的事情是将我们的数据处理成embedding层可输入的形式，这里embedding层的基础概念就不赘述了，需要大家自己了解一下。
大体思路就是：我们是将39列的每一列做为独立的输入之后再合并，所以根据每一列建立索引字典。
建立好之后根据索引字典将raw data（上面的数据视图呈现那样）映射到数字空间，即每个值都代表着索引字典里面的索引，可以根据索引找到原来的值。
对连续型特征变量和分类型特征变量作相应的处理

"""

# There are 13 integer features and 26 categorical features
continous_features = range(1, 14)
categorial_features = range(14, 40)

# Clip integer features. The clip point for each integer feature
# is derived from the 95% quantile of the total values in each feature
continous_clip = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]


# 分类型特征变量处理
class CategoryDictGenerator:
    """
    Generate dictionary for each of the categorical features
    """

    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorial_features, cutoff=0):
        """
        在类别型变量处理时，因为把出现频率太低的数据也加进索引字典的话，会导致模型学习的效果下降，
        所以在建立索引字典的时候我们会将词频太低的数据过滤，
        词频可以通过 cutoff 设置

        """
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):
                    if features[categorial_features[i]] != '':
                        self.dicts[i][features[categorial_features[i]]] += 1
        for i in range(0, self.num_feature):
            self.dicts[i] = filter(lambda x: x[1] >= cutoff,
                                   self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        """
        raw data数据是有存在一些缺少值的，我们对缺失值采取的手段是填0处理

        """
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return [len(self.dicts[idx]) for idx in range(0, self.num_feature)]


# 连续型特征变量处理
class ContinuousFeatureGenerator:
    """
    Clip continuous features.
    """

    def __init__(self, num_feature):
        self.num_feature = num_feature

    def build(self, datafile, continous_features):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')
                for i in range(0, self.num_feature):  # TODO
                    val = features[continous_features[i]]

                    if val != '':
                        val = int(val)
                        if val > continous_clip[i]:
                            val = continous_clip[i]

    # raw data数据是有存在一些缺少值的，我们对缺失值采取的手段是填0处理
    def gen(self, idx, val):
        if val == '':
            return 0.0
        val = float(val)
        return val


def preprocess(datadir, outdir):
    """
    All the 13 integer features are normalzied to continous values and these
    continous features are combined into one vecotr with dimension 13.
    Each of the 26 categorical features are one-hot encoded and all the one-hot
    vectors are combined into one sparse binary vector.
    """
    dists = ContinuousFeatureGenerator(len(continous_features))
    print(os.path.join(datadir, 'train.txt'))
    dists.build(os.path.join(datadir, 'train.txt'), continous_features)

    dicts = CategoryDictGenerator(len(categorial_features))
    dicts.build(
        os.path.join(datadir, 'train.txt'), categorial_features, cutoff=10)

    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [0]
    for i in range(1, len(categorial_features)):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)

    with open(os.path.join(outdir, 'feature_sizes.txt'), 'w') as feature_sizes:
        sizes = [1] * len(continous_features) + dict_sizes
        sizes = [str(i) for i in sizes]
        feature_sizes.write(','.join(sizes))

    random.seed(0)

    # Saving the data used for training.
    with open(os.path.join(outdir, 'train.txt'), 'w') as out_train:
        with open(os.path.join(datadir, 'train.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i]])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                          .rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i]])  # 修改过
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                label = features[0]
                out_train.write(','.join([continous_vals, categorial_vals, label]) + '\n')

    with open(os.path.join(outdir, 'test.txt'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            for line in f:
                features = line.rstrip('\n').split('\t')

                continous_vals = []
                for i in range(0, len(continous_features)):
                    val = dists.gen(i, features[continous_features[i] - 1])
                    continous_vals.append("{0:.6f}".format(val).rstrip('0')
                                          .rstrip('.'))
                categorial_vals = []
                for i in range(0, len(categorial_features)):
                    val = dicts.gen(i, features[categorial_features[i] - 1])  # 修改过
                    categorial_vals.append(str(val))

                continous_vals = ','.join(continous_vals)
                categorial_vals = ','.join(categorial_vals)
                out.write(','.join([continous_vals, categorial_vals]) + '\n')


if __name__ == "__main__":
    preprocess('../data/raw', '../data')

    # 将处理好的数据批量输入DeepFM模型训练

    # 900000 items for training, 10000 items for valid, of all 1000000 items
    Num_train = 800

    # load data
    train_data = CriteoDataset('../data', train=True)
    loader_train = DataLoader(train_data, batch_size=50,
                              sampler=sampler.SubsetRandomSampler(range(Num_train)))
    val_data = CriteoDataset('../data', train=True)
    loader_val = DataLoader(val_data, batch_size=50,
                            sampler=sampler.SubsetRandomSampler(range(Num_train, 899)))

    feature_sizes = np.loadtxt('../data/feature_sizes.txt', delimiter=',')
    feature_sizes = [int(x) for x in feature_sizes]
    print(feature_sizes)

    model = DeepFM(feature_sizes, use_cuda=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.fit(loader_train, loader_val, optimizer, epochs=100, verbose=True)
