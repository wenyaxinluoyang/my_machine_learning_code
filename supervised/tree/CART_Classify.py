'''
CART分类树，基于基尼指数划分
'''

import copy
import numpy as np
from queue import Queue
import pandas as pd

class TreeNode:

    def __init__(self, feat=None, value=None):
        self.feat = feat
        self.split_value = value
        self.left = None
        self.right = None
        self.y = None
        self.gini = None
        self.left_samples = 0
        self.right_samples = 0
        self.total_samples = 0


    def __str__(self):
        return f'feat={self.feat}, split_value={self.split_value}, label={self.y}'

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
             print(feat_name, split_value, gini)
             if gini < min_gini:
                 min_gini = gini
                 best_feat = feat_name
                 best_split_value = split_value
         # print(best_feat, best_split_value, min_gini)
         print('='*10)
         return best_feat, best_split_value, min_gini

    def build_tree(self, data):
        if len(data) == 0: return None
        if len(data[self.target_name].value_counts()) == 1 or len(data) < self.min_samples_split:
            node = TreeNode()
            node.y = data[self.target_name].mode().values[0]
            node.gini = self.gini(data, self.target_name)
            return node
        best_feat, best_split_value, min_gini = self.chose_best_feature(data)
        node = TreeNode(best_feat, best_split_value)
        df1 = data[data[best_feat] == best_split_value]
        df2 = data[data[best_feat] != best_split_value]
        node.left = self.build_tree(df1)
        node.right = self.build_tree(df2)
        node.gini = self.gini(data, self.target_name)
        node.left_samples = len(df1)
        node.right_samples = len(df2)
        node.total_samples = len(data)
        node.y = data[self.target_name].mode().values[0]
        return node

    def fit(self, data):
        # length = len(data)
        # train_size = int(length*0.7)
        # train, test = data[0:train_size], data[train_size: length]
        train = data
        self.target_name = train.columns.tolist()[-1]
        self.feat_names = train.columns.tolist()
        self.feat_names.remove(self.target_name)
        data = copy.copy(train)
        self.root = self.build_tree(data)
        alpha_list, tree_list = self.cut_branch()
        # alpha, tree = self.cross_validation(test, alpha_list, tree_list)


    def cut_branch(self):
        '''
        决策树减枝
        :return:
        '''
        tree_list = [self.root]
        alpha_list = [float('inf')]
        while True:
            # 复制一颗相同额树
            tree = copy.deepcopy(tree_list[-1])
            if self.check(tree): break
            min_alpha, t = float('inf'), None
            node_list = self.bfs(tree)
            # 自下而上对内部节点t计算C(Tt), |Tt|
            for node in node_list:
                leaf_node_list = self.leaf_node(node)
                cost = sum([item.gini for item in leaf_node_list])
                gt = (node.gini - cost)/ (len(leaf_node_list) - 1)
                if gt < min_alpha:
                    min_alpha = gt
                    t = node
            # 减枝, 不需要更新标签，因为我提前在节点里面存放了标签
            t.left = None
            t.right = None
            alpha_list.append(min_alpha)
            self.display(tree)
            print('hahhaha')
            tree_list.append(tree)

        return alpha_list, tree_list

    def cross_validation(self, test, alpha_list, tree_list):
        pass



    def check(self, root):
        if root != None and root.left != None and root.right != None:
            l, r = root.left, root.right
            if l.left==None and l.right==None and r.left==None and r.right==None:
                return True
            else:
                return False
        else:
            return False


    def bfs(self, root):
        '''
        从下到上返回root的内部节点
        :param root: 当前子树
        :return:
        '''
        node = root
        q = Queue(100)
        q.put(node)
        node_list = []
        while q.empty()==False:
            node = q.get()
            # 是叶子节点
            if node.left == None and node.left == None:
                continue
            node_list.append(node)
            if node.left != None: q.put(node.left)
            if node.right != None: q.put(node.right)
        node_list = node_list[1: len(node_list)]
        return node_list[::-1]

    def leaf_node(self, root):
        '''
        统计当前树叶子节点的个数
        :param root: 树根
        :return: 树叶子节点的个数
        '''
        q = Queue(100)
        cnt = 0
        node = root
        q.put(node)
        leaf_node_list = []
        while q.empty()==False:
            node = q.get()
            if node.left == None and node.right == None:
                leaf_node_list.append(node)
            if node.left != None:
                q.put(node.left)
            if node.right != None:
                q.put(node.right)
        return leaf_node_list

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

    def display(self, root):
        print(root.feat, root.split_value, root.y)
        if root.left != None:
            print('left')
            self.display(root.left)
        if root.right != None:
            print('right')
            self.display(root.right)

def load_data():
    df = pd.read_csv('data_set/person.csv')
    # print(df)
    return df

if __name__ == '__main__':
    # l = [1,2,3]
    # print(l[::-1])
    df = load_data()
    model = ClassifyTree()
    model.fit(df)
    # root = model.root
    # model.display(root)
    # s = pd.Series([1,1, 2, 3, 3, 3])
    # print(s.mode().values)

