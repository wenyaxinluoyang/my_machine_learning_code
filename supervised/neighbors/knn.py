'''
K近邻
并使用sklearn的手写数字集做测试

'''

from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from common.distance import *
import copy

def load_data():
    digits = datasets.load_digits()
    images, feature, labels = digits.images, digits.data, digits.target
    return images, feature, labels

class KNeighborsClassifier:
    '''
    K近邻分类
    '''
    def get_dist_func(self, name):
        if name == 'Euclidean':
            return Euclidean().dist
        elif name == 'Manhattan':
            return Manhattan().dist
        elif name == 'Minkowski':
            return Minkowski().dist
        else:
            raise Exception("dist_method 的参数只能从 ['Euclidean', 'Manhattan', 'Minkowski'] 中选择.")

    def __init__(self, n_neighbors=10, algorithm='auto', dist_method='Euclidean', normalize=True):
        '''
        :param n_neighbors: 近邻个数
        :param algorithm: 算法，auto暴力算法，kd_tree
        :param dist: 计算距离的方法，默认是欧式距离
        '''
        self.n_neighbors = n_neighbors
        self.algorithm = 'auto'
        self.get_dist = self.get_dist_func(dist_method)
        self.normalize = normalize
        self.copy = copy

    def build_kd_tree(self, feat, labels):
        '''构建kd树'''
        return None

    def normalize_feat(self):
        m, n = self.feat.shape
        for i in range(n):
            data = self.feat[:, i]
            min_value, max_value = data.min(), data.max()
            if max_value - min_value == 0:
                self.feat[:, i] = 0
            else:
                self.feat[:, i] = (data - min_value) / (max_value - min_value)


    def fit(self, feat, labels):
        self.feat = copy.copy(feat) if self.copy else feat

        # 归一化处理
        if self.normalize:
            self.normalize_feat()

        if self.algorithm == 'auto':
            self.feat = feat
            self.labels = labels
        else:
            # 构建kd树
            root = self.build_kd_tree(feat, labels)

    def predict(self, x_test):
        y_pred = np.array([])
        length = len(x_test)
        for i in range(length):
            y = self.predict_one(x_test[i])
            y_pred = np.append(y_pred, y)
        return y_pred

    def predict_one(self, x):
        '''
        预测x的分类
        :param x: 特征
        :return: 类别
        '''
        if self.algorithm == 'auto':
            return self.auto_predict(x)
        elif self.algorithm == 'kd_tree':
            return self.kd_tree_predict(x)
        else:
            pass
        return None

    def auto_predict(self, x):
        '''
        当算法参数为auto时，暴力求解，计算x到数据集中每个点的距离，取最近的k的分类，遵循多数投票规则
        决出分类
        :param x: 特征
        :return: 预测分类
        '''
        distances = np.array([])
        length = len(self.feat)
        for i in range(length):
            d = self.get_dist(x, self.feat[i])
            distances = np.append(distances, d)
        # 距离从小到大排序
        sorted_distances_index = distances.argsort()
        class_dict = {}
        for i in range(self.n_neighbors):
            index = sorted_distances_index[i]
            label = self.labels[index]
            class_dict[label] = class_dict.get(label, 0) + 1
        label = max(class_dict, key=class_dict.get)
        return int(label)

    def kd_tree_predict(self, x):
        pass

    def score(self, x_test, y_test):
        '''
        计算准确率
        :param x_test:
        :param y_test:
        :return:
        '''
        y_pred = self.predict(x_test)
        accuracy = np.sum(y_pred == y_test) / len(y_pred)
        return accuracy

def show_images(images):
    '''
    绘制前4张图片
    :param images:
    :return:
    '''
    fig, axes = plt.subplots(nrows=1, ncols=4)
    for i in range(4):
        axes[i].imshow(images[i], cmap='gray')
    plt.show()

if __name__ == '__main__':
    # 手写数字数字集
    images, feature, labels = load_data()
    print('样本:', feature.shape)
    x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.2)
    print('训练样本:', x_train.shape, '测试样本:', x_test.shape)
    model = KNeighborsClassifier(n_neighbors=5, normalize=False)
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print('准确率: ', round(accuracy, 2))




    # show_images(images[:4])



