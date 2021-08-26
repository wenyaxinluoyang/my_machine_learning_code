'''
CART分类树，基于基尼指数划分
'''

import copy
import numpy as np

class TreeNode:

    def __init__(self, feat, value):
        self.feat = feat
        self.split_value = value
        self.left = None
        self.right = None
        self.y = None

class ClassifyTree:

    def __init__(self, min_samples_split=2):
        self.min_samples_split = min_samples_split

    def gini(self, data, column_name):
        '''
        计算column_name列的基尼指数
        :param data: 数据集
        :param column_name: 列名
        :return 基尼指数
        '''
        total = len(data)
        g = 1
        for cnt in data[column_name].value_counts():
            prob = cnt / total
            g -= prob**2
        return g

    def feat_best_split_value(self, data, feat_name, target_name):
        '''
        寻找基尼指数最小的划分点
        :param data: 数据
        :param feat_name: 特征名
        :param target_name: 目标名
        :return: feat_name 基尼指数最小的划分值
        '''
        unique_value = set(data[feat_name].values.tolist())
        total = len(data)
        best_split_value, min_gini = None, float('inf')
        for value in unique_value:
            df1 = data[data[feat_name]==value]
            df2 = data[data[feat_name]!=value]
            weight1, weight2 = len(df1)/total, len(df2)/total
            gini = weight1 * self.gini(df1, target_name) + weight2 * self.gini(df2, target_name)
            if gini < min_gini:
                min_gini = gini
                best_split_value = value
        return best_split_value, min_gini


    def chose_best_feature(self, data):
         best_feat, best_split_value, min_gini = None, None, float('inf')
         for feat_name in self.feat_names:
             split_value, gini = self.feat_best_split_value(data, feat_name, self.target_name)
             if gini < min_gini:
                 min_gini = gini
                 best_feat = feat_name
                 best_split_value = split_value
         return best_feat, best_split_value

    def build_tree(self, data):
        if len(data) == 0: return None
        if len(data) < self.min_samples_split:
            node = TreeNode()
            node.y = data[self.target_name].values.mode()
            return node
        best_feat, best_split_value = self.chose_best_feature(data)
        node = TreeNode(best_feat, best_split_value)
        df1 = data[data[best_feat] == best_split_value]
        df2 = data[data[best_feat] != best_split_value]
        node.left = self.build_tree(df1)
        node.right = self.build_tree(df2)
        node.y = data[self.target_name].values.mode()
        return node

    def fit(self, train):
        self.target_name = train.columns.tolist()[-1]
        self.feat_names = train.columns.tolist()
        self.feat_names.remove(self.target_name)
        data = copy.copy(train)
        self.root = self.build_tree(data)
        self.root = self.cut_branch()

    def cut_branch(self):

        return self.root

    def predict(self, test):
        x_test = test[self.feat_names]
        y_pred = []
        for index, row in x_test.iterrows():
            y = self.predict_one(row)
            y_pred.append(y)
        return y_pred

    def predict_one(self, feat_value):
        node = self.root
        result = None
        while node != None:
            result = node.y
            if node.left == None and node.right == None:
                break
            if feat_value[node.feat] == node.split_value:
                node = node.left
            else:
                node = node.right
        return result

    def score(self, test):
        # x_test = test[self.feat_names]
        y_real = test[self.target_name]
        y_pred = self.predict(test)
        accuracy = np.sum(y_real == y_pred)
        return accuracy



if __name__ == '__main__':
    pass
