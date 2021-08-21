'''
感知机简单算法实现(以二分类为例)
模型:
y = sign(w*x + b)
x, y：输入特征, 输出标签
w: 权值向量
b: 偏置
sign: 符合函数

损失函数:
误分类点到分离超平面的距离
L(w, b) = -sum[yi * sign(w*xi+b)]  （xi为误分类点）

学习策略:
每次使用一个误分类点进行随机梯度下降
w = w + learning_rate*yi*xi
b = b + learning_rate*yi
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

class Perceptron:

    def __init__(self, learning_rate=0.1):
        self.weights = None
        self.bias = None
        self.learning_rate = learning_rate

    def sign(self, t):
        if t > 0: return 1
        else: return -1

    def func(self, x):
        y = self.sign(np.dot(self.weights, x) + self.bias)
        return y

    def fit(self, feat, labels):
        '''
        感知机原始形式，求解模型参数
        :param feat: 特征
        :param labels: 标签
        :return:
        '''
        row, col = feat.shape
        # 初始化权值向量和偏置, (row, 1)
        self.weights, self.bias = np.zeros(col), 0
        cnt = 0
        while True:
            flag = True
            cnt = cnt + 1
            for i in range(row):
                # 选取误分类的点，进行随机梯度下降
                x, y = feat[i], labels[i]
                result = self.func(x)
                if y * result <= 0:
                    self.SDG(x, y)
                    flag = False
            if flag: break
        print('='*20, f'共迭代{cnt}次', '='*20)

    def SDG(self, x, y):
        '''
        随机梯度下降
        :param x: 误分类点的特征
        :param y: 误分类点的标签
        :return:
        '''
        self.weights += self.learning_rate*y*x
        self.bias += self.learning_rate*y

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

    # 求解模型参数
    model = Perceptron()
    model.fit(feature, labels)
    weights, bias = model.weights, model.bias
    print(weights, bias)

    # 显示数据和求得的分离超平面
    display(feature, labels, weights, bias)




