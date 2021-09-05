# 网格调参
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn import model_selection, metrics
from matplotlib import pyplot as plt

def get_model(feature, labels):
    # x_train, x_test, y_train, y_test = train_test_split(feature, labels, test_size=0.3, random_state=0)
    # dtrain = xgb.DMatrix(x_train, label=y_train)
    # dtest = xgb.DMatrix(x_test, label=y_test)
    dtrain = xgb.DMatrix(feature, labels)

    default_param = {'silent': True
        , 'obj': 'binary:logistic'
        , 'eval': 'auc'
        , "subsample": 1
        , "max_depth": 6
        , "eta": 0.3
        , "gamma": 0
        , "lambda": 1
        , "alpha": 0
        , "colsample_bytree": 1
        , "colsample_bylevel": 1
        , "colsample_bynode": 1
        , "nfold": 5}

    num_round = 50

    cv_result = xgb.cv(default_param, dtrain, num_round)
    fig, ax = plt.subplots(1, figsize=(15, 8))
    ax.set_ylim(top=5)
    ax.grid()
    ax.plot(range(1, 201), cv_result.iloc[:, 0], c="red", label="train,original")
    ax.plot(range(1, 201), cv_result.iloc[:, 2], c="orange", label="test,original")
    ax.legend(fontsize="xx-large")
    # plt.show()
    cv_params = {
        'eta': np.arange(0.01, 0.2, 0.02),
        'max_depth': range(3, 11),
    }

