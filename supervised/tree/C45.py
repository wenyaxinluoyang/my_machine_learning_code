'''
C45构建决策树
'''

'''
ID3算法构建决策树， 使用信息增益比
'''
import copy
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class TreeNode:
    def __init__(self, feat_name=None, feat_value=None, label=None):
        '''
        节点
        :param feat_name: 特征名称
        :param feat_value: 特征值
        '''
        self.feat_name = feat_name
        self.feat_value = feat_value
        self.child_list = None
        self.label = label

    def __str__(self):
        result = f"特征名={self.feat_name}, 特征值={self.feat_value}, 标签={self.label}"
        return result

class DecisionTreeClassifier:

    def __init__(self, threshold=0):
        self.threshold = threshold

    def entropy(self, data, column_name):
        '''
        计算熵
        :param data: 参数类型必须是DataFrame
        :param column_name: 要计算熵的列名
        :return:
        '''
        n_samples = len(data)
        ent = 0
        for cnt in data[column_name].value_counts():
            prob = cnt / n_samples
            ent -= prob * math.log(prob, 2)
        return ent

    def infor_gain(self, base_ent, data, feature_name, target_name):
        '''
        计算信息增益增益比
        :param base_ent: 不划分时的熵
        :param data: 数据集, 要求类型是DataFrame
        :param feature_name: 要计算信息增益的特征名
        :param target_name: 目标变量的名称
        :return: 信息增益比
        '''
        n_samples = len(data)
        ent = 0
        values = data[feature_name].unique().tolist()
        for value in values:
            temp_data = data[data[feature_name] == value]
            weight = len(temp_data) / n_samples
            ent += weight * self.entropy(temp_data, target_name)
        gain = base_ent - ent
        return gain

    def chose_best_feature(self, data):
        '''
        获取最优特征
        :param data: DataFrame 由特征和目标组成，目标必须放在最后一行
        :return: 最优特征，最大的信息增益
        '''
        columns = data.columns.tolist()
        target_name = columns[-1]
        feature_name = columns[0: -1]
        max_gain_ratio, best_feature = 0, feature_name[0]
        base_ent = self.entropy(data, target_name)
        for name in feature_name:
            gain_ratio = self.infor_gain(base_ent, data, name, target_name)/self.entropy(data, name)
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                best_feature = name
            # print('best_feature=', name, '   best_gain=', gain_ratio)
        return best_feature, max_gain_ratio

    def build_tree(self, data):
        '''
        ID3构建决策树, 按照信息增益进行划分。
        :return:
        '''
        columns = data.columns.tolist()
        target_name = columns[-1]
        feature_name = columns[0: -1]
        if len(data[target_name].value_counts()) == 1:
            label = data[target_name].mode().values
            leaf_node = TreeNode(label=label)
            return leaf_node
        if len(feature_name) == 0:
            label = data[target_name].mode().values
            leaf_node = TreeNode(label=label)
            return leaf_node
        best_feature, max_gain = self.chose_best_feature(data)
        # print('='*50)
        # 最大信息增益小于阈值，停止分列
        if max_gain < self.threshold:
            label = data[target_name].mode().values
            leaf_node = TreeNode(label=label)
            return leaf_node
        node = TreeNode(feat_name=best_feature)
        values = data[best_feature].unique().tolist()
        for value in values:
            temp_data = data[data[best_feature] == value]
            temp_data = temp_data.drop(best_feature, axis=1)
            child = self.build_tree(temp_data)
            child.feat_name = best_feature
            child.feat_value = value
            if node.child_list is None: node.child_list = []
            node.child_list.append(child)
        return node

    def fit(self, data):
        '''
        使用ID3算法构建决策树
        :param data: 参数类型是DataFrame
        :return:
        '''
        d = copy.copy(data)
        self.columns = d.columns.tolist()[0:-1]
        self.root = self.build_tree(d)

    def predict(self, x_test):
        '''
        预测测试集的分类
        :param x_test: 必须是DataFrame，特征需要和训练集保持一致
        :return: 预测值，是一个1维的ndarry
        '''
        columns = sorted(x_test.columns.tolist())
        self.columns = sorted(self.columns)
        flag = False
        for train_feat_name, test_feat_name in zip(self.columns, columns):
            if train_feat_name != test_feat_name:
                flag = True
                break
        if flag:
            raise Exception('训练集和测试集的特征不相同')
        y_pred = np.array([])
        for index, row in x_test.iterrows():
            dic = {}
            for col in columns:
                dic[col] = row[col]
            y = self.predict_one(self.root, dic)
            y_pred = np.append(y_pred, y)
        return y_pred


    def predict_one(self, node, dic):
        '''
        预测一个输入
        :param node: 根节点
        :param dic: 输入
        :return: 输出
        '''
        y = None
        while True:
            child_list = node.child_list
            if child_list == None:
                y = node.label
                break
            flag = True
            for child in child_list:
                if child.feat_value == dic[child.feat_name]:
                    node = child
                    flag = False
                    break
            if flag:
                raise Exception('包含不合法的特征值')
        return y

    def display_tree(self):
        self.display(self.root)

    def display(self, node):
        print(node)
        if node.child_list is not None:
            for child in node.child_list:
                self.display(child)

    def score(self, x_test, y_test):
        '''
        计算测试集的准确率
        :param x_test: 测试集
        :param y_test: 测试集的真实目标值
        :return: 准确率
        '''
        y_pred = self.predict(x_test)
        accuracy = sum(y_pred == y_test) / len(y_test)
        return accuracy


if __name__ == '__main__':
    data = pd.read_csv('data_set/data.csv')
    model = DecisionTreeClassifier()
    model.fit(data)
    # root = model.root
    model.display_tree()

    x_test = pd.DataFrame({'hair': ['long', 'long'], 'voice': ['thin', 'thick'],  'height': ['<172.0', '>172.0'], 'ear_stud': ['yes', 'no']})
    y_pred = model.predict(x_test)
    print(y_pred)
    # columns = ['A', 'B', 'C']
    # print(columns[0: -1])
    # arr = np.array([[1, 2, 3], [3, 2, 3]])
    # df = pd.DataFrame(arr)
    # print(df)