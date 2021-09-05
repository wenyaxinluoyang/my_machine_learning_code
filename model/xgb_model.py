'''
使用xgboost来建模
参数
booster: gbtree, gblinear, dart, gbtree和cart使用树基础分类器，gblinear使用线性分类器
nthread: 模型可用的最大线程数
eta: 别名：learning_rate，学习速率，范围[0,1]
gamma: 在树的叶节点上进行进一步分区所需的最小损失减少，越大gamma，算法越保守，范围[0,inf)
max_depth: 默认为6，一棵树的最大深度
min_child_weight: 汉字需要的最西奥实例权重总和(hessian)，如果树分区步骤导致实例权重总和小于叶节点的min_child_weight，放弃分列
subsample 训练实例的子样本比例，将其设置为0.5，xgboost在训练树木之前随机采样一般的训练数据，
colsample_bytree, colsample_bylevel, colsample_bynode，默认值为1，用于对列进行二次采样的一系列参数
lambda 默认为1，权重的L2正则化项
alpha 权重的L!正则化项
objective: reg:squarederror平方损失回归，reg:squarederror回归平方对数损失,reg: logistic逻辑回归, reg:pseudohubererror，huber损失
binary:logistic 二元分类的逻辑回归，输出概率
binary: logitraw 二元分类逻辑回归，逻辑变换前的输出分数
binary: hinge 二元分类的教练损失，这使得预测为0或1，而不是产生概率。
eval_metric: 验证数据的评估指标

params = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'gamma': 0.1,
    'max_depth': 8,
    'alpha': 0,
    'lambda': 0,
    'subsample': 0.7,
    'colsample_bytree': 0.5,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.03,
    'nthread': -1,
    'seed': 2019,
}
'''
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def encode_cate_feature(df, cate_feature):
    cate_feat_df = pd.get_dummies(df[cate_feature])
    data = pd.concat([df, cate_feat_df], axis=1)
    data.drop(cate_feature, axis=1, inplace=True)
    return data

def get_model(feature, labels, cate_feature):
    data = encode_cate_feature(feature, cate_feature)
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0)
    params = {
        'objective': 'binary:logistic',
        'max_depth': 4,
        'eta': 0.1,
        'nthread': 2,
        'eval_metric': 'auc'
    }
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_test = xgb.DMatrix(x_test, labels=y_test)
    evalist = [(d_train, 'train', d_test, 'test')]
    model = xgb.train(params, d_train, 100, evalist)
    y_pred = model.predict(d_test)
    return model

if __name__ == '__main__':
    df = pd.read_csv('../supervised/tree/data_set/person.csv')
    cate_feature = df.columns.tolist()
    cate_feature.remove('labels')
    print(df.head())
    result = pd.get_dummies(df[cate_feature])
    data = pd.concat([df, result], axis=1)
    data.drop(cate_feature, axis=1, inplace=True)
    print(data.head())




