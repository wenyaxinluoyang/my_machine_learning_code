'''
数据预处理

missing_rate 计算列/样本的缺失率
missing_rate_bin 按照缺失率分箱，比如看缺失率在(0.9,1]的变量占总变量的个数
missing_rate_bar 绘制缺失率分箱后的条形图
delete_by_threshold 把缺失率大于指定阈值的列(axis=0)/样本(axis=1)删除
cate_fill_nan 离散型变量填充

'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def missing_rate(df, axis=0):
    '''
    计算各列的缺失率
    :param df: DataFrame
    :return: 表中各列的缺失率
    '''
    if axis == 0:
        result = df.isnull().sum(axis=axis)/df.shape[0]
        columns = ['column', 'missing_rate']
    else:
        result = df.isnull().sum(axis=axis)/df.shape[1]
        columns = ['sample_index', 'missing_rate']
    result = result.reset_index()
    result.columns = columns
    result['missing_rate'] = result['missing_rate'].round(2)
    result.sort_values(by=['missing_rate'], ascending=False, inplace=True)
    note = dict()
    note['缺失'] = result[result['missing_rate'] != 0].shape[0]
    note['无缺失'] = result.shape[0] - note['缺失']
    note['占比'] = round(note['缺失'] / result.shape[0], 2)
    return note, result

def missing_rate_bin(df, axis=0):
    '''
    缺失率的分布
    :param df:
    :param axis:
    :return: 缺失率在某个区间的列数占总列数占比，样本数占总样本数占比
    '''
    note, result = missing_rate(df, axis)
    bins = np.arange(0, 1.1, 0.1)
    # 缺失率>0的列数或样本数
    total = result[result['missing_rate'] > 0].shape[0]
    missing_rate_distributed = result['missing_rate'].groupby(pd.cut(result['missing_rate'], bins)).count()/total
    missing_rate_distributed = missing_rate_distributed.round(2)
    rate_list = [value for value in missing_rate_distributed]
    dic = dict()
    for i, rate in enumerate(rate_list):
        v = round(bins[i], 1)
        bin_str = '(' + str(v) + ', ' + str(round(v+0.1, 1)) + ']'
        # print(bin_str, rate)
        dic[bin_str] = rate
    # print(dic)
    return dic

def missing_rate_bar(df, axis=0, figsize=None):
    '''
    缺失率分布图
    :param df: DataFrame
    :param axis: 0 按列统计缺失率，1 按样本统计缺失率
    :param figsize: 分布图大小
    :return: 缺失率分布图，
    '''
    # note, result = missing_rate(df, axis)
    # result = result[result['missing_rate'] > 0]
    result = missing_rate_bin(df, axis)
    plt.figure(figsize=figsize)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    x_list = [i for i in range(0, 10, 1)]
    y_list = [v for k,v in result.items()]
    label_list = [k for k,v in result.items()]
    plt.bar(x_list, y_list, color='#0066CC', alpha=0.75)
    for x, y in zip(x_list, y_list):
        if y == 0: continue
        plt.text(x-0.2, y/2, str(y))
    # plt.xticks(bins)
    plt.xticks(x_list, labels=label_list)
    plt.title('缺失率柱状图')
    plt.xlabel('缺失率')
    plt.ylabel('占比')
    plt.grid(True, alpha=0.3)
    return plt.show()

def delete_by_threshold(df, axis, threshold=None):
    data = df.copy()
    note, result = missing_rate(df, axis)
    tmp = result[result['missing_rate'] > threshold]

    # 删除缺失值大于阈值的列
    if axis == 0:
        print('缺失率超过{}的列个数为{}'.format(threshold, tmp.shape[0]))
        drop_columns = tmp['column'].values.tolist()
        data.drop(drop_columns, axis=1, inplace=True)
        return data
    else:
        print('缺失率超过{}的样本个数为{}'.format(threshold, tmp.shape[0]))
        drop_index = tmp['sample_index'].values.tolist()
        data.drop(drop_index, axis=0, inplace=True)
        return data





if __name__ == '__main__':
    data = pd.DataFrame()
    data['x'] = [1, 2, 3, 4, None, 5]
    data['y'] = [None, 2, 3, None, 4, 5]
    data['z'] = [None, 2, None, 3, 4, 5]
    # print(data)
    # print('='*10)
    # missing_rate(data)
    # hist_missing_rate(data, axis=1, figsize=(8, 4))
    # missing_rate_bin(data)
    # missing_rate_bar(data, axis=0, figsize=(8,4))
    # note, result = missing_rate(data, axis=0)
    # print(result)
    # df = delete_by_threshold(data, axis=0, threshold=0.3)
    # print(df)


