'''
CART
回归树： 最小二乘回归树
'''

import copy
import numpy as np

class TreeNode:
    def __init__(self, feat=None, split_value=None, left=None, right=None, y=None):
        self.feat = feat
        self.split_value = split_value
        self.left = None
        self.right = None
        self.y = None

class RegressionTree:

    def feat_best_split_point(self, data, feat_name, target_name):
        '''
        计算特征的最佳分割点，效果以分割后的平方和为结果
        :param data: 数据集，DataFrame
        :param feat_name: 特征名
        :param target_name: 目标变量名
        :return:
        '''
        # data[feat_name] = data[feat_name].astype(np.float64)
        values = data[feat_name].values
        unique_value = set(values.tolist())
        best_point, min_s = None, float('inf')
        for value in unique_value:
            left = data[data[feat_name] <= value][target_name].values
            right = data[data[feat_name] > value][target_name].values
            left_mean, right_mean = left.mean(), right.mean()
            s = np.sum(np.square(left - left_mean)) + np.sum(np.square(right - right_mean))
            if s < min_s:
                min_s = s
                best_point = value
        return min_s, best_point

    def chose_best_feature(self, data, target_name):
        '''
        寻找最优划分特征，及相应的划分点。
        :param data: 数据集
        :param target_name: 目标变量名
        :return: 最优特征名，相应的划分点
        '''
        feat_names = data.columns.tolist()
        feat_names.remove(target_name)
        best_feat, feat_best_point, best_min_s = None, None, float('inf')
        for feat_name in feat_names:
            min_s, best_point = self.feat_best_split_point(data, feat_name, target_name)
            if min_s < best_min_s:
                best_min_s = min_s
                best_feat = feat_name
                feat_best_point = best_point
        return best_feat, feat_best_point

    def __init__(self, min_samples_split=2):
        '''

        :param min_samples_split: 分裂一个节点所需要的最小样本数
        '''
        self.min_samples_split = min_samples_split

    def build_tree(self, data, target_name):
        if len(data) < self.min_samples_split:
            result = data[target_name].values.mean()
            node = TreeNode(y=result)
            return node
        best_feat, feat_best_point = self.chose_best_feature(data, target_name)
        node = TreeNode(best_feat, feat_best_point)
        left_data = data[data[best_feat] <= feat_best_point]
        right_data = data[data[best_feat] > feat_best_point]
        # 递归构建左右子树，默认左子数存放比feat_best_point小的数据，模型右子数存放比feat_best_point大的数据
        node.left = self.build_tree(left_data, target_name)
        node.right = self.build_tree(right_data, target_name)
        node.y = data[target_name].values.mean()
        return node

    def fit(self, train):
        data = copy.copy(train)
        # 默认最后一列为目标值
        self.feat_name = data.columns.tolist()
        self.target_name = data.columns.tolist[-1]
        self.feat_name.remove(self.target_name)
        # 建树
        self.root = self.build_tree(data, self.target_name)
        # 减枝
        self.root = self.cut_branch()


    def cut_branch(self):
        '''决策树减枝'''
        return self.root

    def predict(self, test):
        '''
        给出测试集的预测结果
        :param x_test: 测试集
        :return: 测试集的预测结果
        '''
        y_pred = []
        # y_test = test[self.target_name]
        x_test = test[self.feat_name]
        for index, row in x_test.iterrows():
            y = self.predict_one(row)
            y_pred.append(y)
        return np.array(y_pred)


    def predict_one(self, feat_value):
        '''
        给出一个样本的预测
        :param feat_value: 特征及其值
        :return: 预测值
        '''
        node = self.root
        result = None
        while node != None:
            # 叶子节点返回
            result = node.y
            if node.left == None and node.right == None:
                break
            if feat_value[node.feat] <= node.split_value:
                node = node.left
            else:
                node = node.right
        return result


    def score(self, test):
        '''
        输出测试集的平方误差
        :param test: 测试集，包含特征和目标
        :return: 平均平方误差
        '''
        y_pred = self.predict(test)
        y_real = test[self.target_name]
        mse = np.sum(np.square(y_real - y_pred)) / len(y_pred)
        return mse


if __name__ == '__main__':
    pass