'''
变量分箱
chi_merge 卡方分箱
binning_num 连续型变量分箱
binning_cate 离散型变量分箱
binning_self 自定义分箱
'''

import math
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def cal_chi2(bin1, bin2, all_bad_rate):
    '''
    计算两个箱的卡方值
    :param bin1: 箱1
    :param bin2: 箱2
    :param all_bad_rate: 整体坏用户概率
    :return: 卡方值
    '''
    bin1_bad_exp, bin1_good_exp = bin1['total']*all_bad_rate, bin1['total']*(1-all_bad_rate)
    bin2_bad_exp, bin2_good_exp = bin2['total']*all_bad_rate, bin2['total']*(1-all_bad_rate)
    x = (bin1['bad']-bin1_bad_exp)**2/bin1_bad_exp + (bin1['total']-bin1['bad']-bin1_good_exp)**2/bin1_good_exp
    y = (bin2['bad']-bin2_bad_exp)**2/bin2_bad_exp + (bin2['total']-bin2['bad']-bin2_good_exp)**2/bin2_good_exp
    return x+y


def merge_bin(bin_dict_list, index1, index2):
    '''
    合并两个箱，把索引为index2的箱合并到index1上，并删除index2
    :param bin_dict_list: 箱列表
    :param index1: 索引1
    :param index2: 索引2
    :return: 合并后的列表
    '''
    # 把index2合并到index1s上，并删除index2
    bin_dict_list[index1]['total'] = bin_dict_list[index1]['total'] + bin_dict_list[index2]['total']
    bin_dict_list[index1]['bad'] = bin_dict_list[index1]['bad'] + bin_dict_list[index2]['bad']
    bin_dict_list[index1]['bad_rate'] = bin_dict_list[index1]['bad'] / bin_dict_list[index1]['total']
    bin_dict_list[index1]['cut_point'] = max(bin_dict_list[index1]['cut_point'], bin_dict_list[index2]['cut_point'])
    bin_dict_list.pop(index2)
    return bin_dict_list

def init_bin_dict_list(df, col, target, cut_points):
    bin_dict_list = []
    length = len(cut_points)
    for i in range(length - 1):
        temp_df = df[(df[col] > cut_points[i]) & (df[col] <= cut_points[i + 1])]
        bin_dict = {}
        bin_dict['cut_point'] = cut_points[i + 1]
        bin_dict['total'] = len(temp_df)
        bin_dict['bad'] = sum(temp_df[target])
        bin_dict['bad_rate'] = bin_dict['bad'] / bin_dict['total']
        bin_dict_list.append(bin_dict)
    return bin_dict_list


def chi_merge(df, col, target, max_bin, min_pct):
    '''
    卡方分箱
    :param df: 数据
    :param col: 要进行分箱的列名
    :param target: 目标变量名
    :param max_bin: 箱数
    :param min_pct: 箱内样本占总样本的最小比例
    :return: 切割点
    '''
    values = sorted(df[col].values.tolist())
    unique_values = list(set(values))
    total = len(df)
    bin_dict_list = []
    all_bad_rate = sum(df[target]) / total #df[target].sum() / total
    if min_pct > 0:
        step = math.floor(total * min_pct)
        num = math.floor(total / step)
        cut_points = [values[i*step] for i in range(num)]
        cut_points.append(float('inf'))
        cut_points = sorted(list(set(cut_points)))
        cut_points[0] = cut_points[0]-1
        bin_dict_list = init_bin_dict_list(df, col, target, cut_points)
    elif len(unique_values) > 100:
        step = math.floor(total / 100)
        cut_points = [values[i*step] for i in range(100)]
        cut_points.append(float('inf'))
        cut_points = sorted(list(set(cut_points)))
        cut_points[0] = cut_points[0] - 1
        bin_dict_list = init_bin_dict_list(df, col, target, cut_points)
    else:
        for value in sorted(unique_values):
            temp_df = df[df[col] == value]
            bin_dict = {}
            bin_dict['cut_point'] = value
            bin_dict['total'] = len(temp_df)
            bin_dict['bad'] = sum(temp_df[target])#temp_df[target].sum()
            bin_dict['bad_rate'] = bin_dict['bad']/bin_dict['total']
            bin_dict_list.append(bin_dict)

    while len(bin_dict_list) > max_bin:
        length = len(bin_dict_list)
        min_chi, index = float('inf'), 0
        for i in range(length-1):
            chi = cal_chi2(bin_dict_list[i], bin_dict_list[i+1], all_bad_rate)
            if chi < min_chi:
                min_chi = chi
                index = i
        # 合并
        bin_dict_list = merge_bin(bin_dict_list, index, index+1)
    # cut_points = [bin['cut_point'] for bin in bin_dict_list]

    while True:
        length, index = len(bin_dict_list), -1
        for i in range(length):
            bin = bin_dict_list[i]
            if bin['bad'] == 0 or bin['bad'] == bin['total']:
                index = i
                break
        if index == -1: break
        bin_dict_list = deal_bin(bin_dict_list, index, all_bad_rate)

    # if min_pct > 0:
    #     while True:
    #         length, index = len(bin_dict_list), -1
    #         for i in range(length):
    #             bin = bin_dict_list[i]
    #             if (bin['total'] / total) < min_pct:
    #                 index = i
    #                 break
    #         if index == -1: break
    #         bin = bin_dict_list[index]
    #         print(bin['total'] / total, min_pct)
    #         bin_dict_list = deal_bin(bin_dict_list, index, all_bad_rate)

    cut_points = [bin['cut_point'] for bin in bin_dict_list]
    cut_points = [values[0]-0.1] + cut_points
    cut_points[-1] = values[-1] + 0.1
    return cut_points

def deal_bin(bin_dict_list, index, all_bad_rate):
    '''
    处理全是好用户或坏用户的箱
    :param bin_dict_list:
    :param index:
    :param all_bad_rate:
    :return:
    '''
    length = len(bin_dict_list)
    if index == 0:
        # 删除第一个箱
        bin_dict_list = merge_bin(bin_dict_list, index + 1, index)
    elif index == length - 1:
        # 删除最后一箱
        bin_dict_list = merge_bin(bin_dict_list, index - 1, index)
    else:
        bin = bin_dict_list[index]
        pre_bin = bin_dict_list[index - 1]
        nex_bin = bin_dict_list[index + 1]
        chi_pre = cal_chi2(pre_bin, bin, all_bad_rate)
        chi_nex = cal_chi2(bin, nex_bin, all_bad_rate)
        if chi_pre < chi_nex:
            bin_dict_list = merge_bin(bin_dict_list, index - 1, index)
        else:
            bin_dict_list = merge_bin(bin_dict_list, index, index + 1)
    return bin_dict_list


def binning_self(df, col, target, bins, right_border=True):
    '''
    :param df: 数据
    :param col: 单个要进行分箱的变量名
    :param target: 目标变量
    :param bins: 划分区间的list
    :param right_border: 设定左开右闭、左闭右开
    :return:
    '''
    total = len(df)
    total_bad = df[target].sum()
    total_good = total - total_bad
    odds = total_good / total_bad
    bucket = pd.cut(df[col], bins, right=right_border)
    d1 = df.groupby(bucket)
    result = pd.DataFrame()
    result['min'] = d1[col].min()
    result['max'] = d1[col].max()
    result['total'] = d1[target].count()
    result['total_rate'] = result['total']/total
    result['bad'] = d1[target].sum()
    result['bad_rate'] = result['bad'] / result['total']
    result['good'] = result['total'] - result['bad']
    result['good_rate'] = result['good'] / result['total']
    result['bad_attr'] = result['bad'] / total_bad
    result['good_attr'] = result['good'] / total_good
    result['odds'] = result['good'] / result['bad']
    gb_list = []
    for value in result.odds:
        if value >= odds:
            gb_index = str(round(value/odds*100, 0)) + str('G')
        else:
            gb_index = str(round(odds/value*100, 0)) + str('B')
        gb_list.append(gb_index)
    result['GB_index'] = gb_list
    result['woe'] = np.log(result['good_attr']/result['bad_attr'])
    result['bin_iv'] = (result['good_attr'] - result['bad_attr']) * result['woe']
    result['IV'] = result['bin_iv'].sum()
    iv = result['bin_iv'].sum().round(3)
    print('变量名:{}'.format(col))
    print('IV:{}'.format(iv))
    bin_df = result.copy()
    return bin_df, iv

def binning_cate(df, target, col_list):
    '''
    离散型变量分箱
    :param df: 数据
    :param target: 目标变量
    :param col_list: 离散型变量列表
    :return: 分箱
    '''
    total = len(df)
    total_bad = sum(df[target])
    total_good = total - total_bad
    total_odds = total_good / total_bad
    bin_df_list = []
    iv_list = []
    for col in col_list:
        groups = df.groupby([col])
        bin_df = pd.DataFrame()
        bin_df['total'] = groups[target].count()
        bin_df['bad'] = groups[target].sum()
        bin_df['good'] = bin_df['total'] - bin_df['bad']
        bin_df['bad_rate'] = bin_df['bad'] / bin_df['total']
        bin_df['good_rate'] = bin_df['good'] / bin_df['total']
        bin_df['total_rate'] = bin_df['total'] / total
        bin_df['bad_attr'] = bin_df['bad'] / total_bad
        bin_df['good_attr'] = bin_df['good'] / total_good
        bin_df['odds'] = bin_df['good'] / bin_df['bad']
        bin_df['woe'] = np.log(bin_df['good_attr'] / bin_df['bad_attr'])
        bin_df['bin_iv'] = (bin_df['good_attr'] - bin_df['bad_attr']) * bin_df['woe']
        bin_df['iv'] = sum(bin_df['bin_iv'])
        gb_list = []
        for value in bin_df.odds:
            if value >= total_odds:
                gb_index = str(round(value / total_odds * 100, 0)) + str('G')
            else:
                gb_index = str(round(total_odds / value * 100, 0)) + str('B')
            gb_list.append(gb_index)
        bin_df['gb_index'] = gb_index
        bin_df = bin_df.reset_index()
        bin_df = bin_df.rename(columns={col: 'bin'})
        iv = sum(bin_df['bin_iv'])
        bin_df_list.append(bin_df)
        iv_list.append(iv)

    return bin_df_list, iv_list



def binning_num(df, target, col_list, max_bin=5, min_pct=0.05, algorithm='chi'):
    '''
    对col_list这些列，进行卡方分箱
    :param df: 数据集
    :param target: 目标变量字段名
    :param col_list: 要分箱的连续型变量集合
    :param max_bin: 最大的分箱个数
    :param min_binpct: 区间内样本所在总体的最小化
    :return: bin_df_list, 里面存储每个变量的分箱结果, iv_value，里面存储每个变量的IV值
    '''
    total = len(df)
    # total_bad = df[target].sum()
    # total_good = total - total_bad
    # total_odds = total_good/total_bad
    # inf, ninf = float('inf'), float('-inf')
    bin_df_list = []
    iv_list = []
    for col in col_list:
        if algorithm == 'chi':
            bins = chi_merge(df, col, target, max_bin, min_pct)
            print(bins)
        elif algorithm == 'tree':
            bins = decision_tree_binning(df, col, target, max_bin, min_pct)
            print(bins)
        else:
            raise Exception('参数错误，只能选择（chi）卡方分箱或决策树分箱(tree)')
        bin_df, iv = binning_self(df, col, target, bins, right_border=True)
        bin_df_list.append(bin_df)
        iv_list.append(iv)
    return bin_df_list, iv_list

def decision_tree_binning(df, col, target, max_leaf_nodes=6, min_samples_leaf=0.05, nan_value=-999):
    cut_points = []
    x = df[col].fillna(nan_value).values
    y = df[target].values
    clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)
    clf.fit(x.reshape(-1, 1), y)
    node_cnt = clf.tree_.node_count
    child_left = clf.tree_.children_left
    child_right = clf.tree_.children_right
    threshold = clf.tree_.threshold
    for i in range(node_cnt):
        if child_left[i] != child_right[i]:
            cut_points.append(threshold[i])
    min_x = x.min() - 0.1
    max_x = x.max() + 0.1
    cut_points.append(min_x)
    cut_points.append(max_x)
    cut_points.sort()
    return cut_points


if __name__ == '__main__':
    df = pd.read_csv('../binning/mini_data.csv')
    binning_num(df, 'label', ['mz_score'], max_bin=5, min_pct=0.05, algorithm='tree')
    # df = pd.read_csv('mini_data.csv')
    # cut_points = chi_merge(df, 'mz_score', 'label', max_bin=5, min_pct=0.05)
    # print(cut_points)
    # path = "E:\python_code\my_machine_learning_code\supervised\tree\data_set\person.csv"
    # df = pd.read_csv('../supervised/tree/data_set/person.csv')
    # df['labels'] = df['labels'].replace({'是': 0, '否': 1})
    # print(df)
    # cate_var_list = ['年龄', '有工作', '有自己的房子', '信贷情况']
    # bin_df_list, iv_list = binning_cate(df, 'labels', cate_var_list)
    # for col, bin_df, iv in zip(cate_var_list, bin_df_list, iv_list):
    #     print(col, 'IV:', iv)
    #     print(bin_df[['bin', 'bad_rate', 'woe', 'iv']])
    #     print('='*100)