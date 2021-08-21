'''
感知机的对偶算法

模型
y = sign(sum(aj*yj*xj)*x + b)
aj = learning_rate * nj, nj为第j个点的修改次数

aj = aj + learning_rate
b = b + learning_rate * yj
'''

import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gen_data():
    '''
    生成线性可分的数据，以3维空间为例
    :return: 线性可分的数据及其标签
    '''
    feature, labels = datasets.make_blobs(n_samples=100, n_features=3, centers=[(-20, -17, -15), (0, 20, 23)], cluster_std=[9, 9])
    labels[labels == 0] = -1
    return feature, labels

class DualPerceptron:

    def __init__(self, learning_rate=0.1):
        '''设置模型的一些参数'''
        self.learning_rate = learning_rate

    def cal_gram(self, feat):
        '''
        计算gram矩阵
        :param feat: 特征
        :return:
        '''
        row = feat.shape[0]
        gram = [[0 for i in range(row)] for j in range(row)]
        for i in range(row):
            for j in range(i, row):
                gram[i][j] = np.dot(feat[i], feat[j])
                gram[j][i] = gram[i][j]
        self.gram = gram

    def sign(self, t):
        if t > 0: return 1
        else: return -1

    def func(self, x):
        return self.sign(np.dot(self.weights, x) + self.bias)

    def fit(self, feat, labels):
        row, col = feat.shape
        # self.cal_gram(feat)
        self.alphas, self.bias = np.zeros(row), 0
        cnt = 0
        while True:
            flag = True
            cnt = cnt + 1
            for i in range(row):
                x, y = feat[i], labels[i]
                self.weights = np.dot((self.alphas * labels).T, feat)
                if y * self.func(x) <= 0:
                    self.alphas[i] += self.learning_rate
                    self.bias +=  self.learning_rate * y
                    flag = False
            if flag: break
        print('=' * 20, f'共迭代{cnt}次', '=' * 20)

def display(feature, labels, weights, bias):
    '''
    可视化
    :param feature: 特征
    :param labels: 标签
    :param weights: 权值向量
    :param bias: 偏置
    :return:
    '''
    pos_index = list(np.argwhere(labels==1).ravel())
    neg_index = list(np.argwhere(labels==-1).ravel())
    pos_data = feature[pos_index, :]
    neg_data = feature[neg_index, :]

    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(pos_data[:, 0], pos_data[:, 1], pos_data[:, 2], color='blue', marker='*')
    ax.scatter(neg_data[:, 0], neg_data[:, 1], neg_data[:, 2], color='red', marker='^')
    font_dict = {'size': 15}
    ax.set_xlabel('X', fontdict=font_dict)
    ax.set_ylabel('Y', fontdict=font_dict)
    ax.set_zlabel('Z', fontdict=font_dict)

    # 绘制平面
    x_min, x_max = feature[:, 0].min() - 1, feature[:, 0].max() + 1
    y_min, y_max = feature[:, 1].min() - 1, feature[:, 1].max() + 1
    x,  y = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    z = (-bias - weights[0]*x - weights[1]*y)/weights[2]
    ax.plot_surface(x, y, z, lw=0, antialiased=False, color='gray', alpha=0.2)
    plt.show()

if __name__ == '__main__':
    # 生成线性可分的数据集
    feature, labels = gen_data()
    # 感知机对偶算法，求解模型参数
    model = DualPerceptron()
    model.fit(feature, labels)
    # 可视化
    display(feature, labels, model.weights, model.bias)


