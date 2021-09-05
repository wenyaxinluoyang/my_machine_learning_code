'''
常用损失函数
'''
import math
import numpy as np


def zero_one_loss(y_real, y_pred):
    '''0-1损失函数'''
    if y_real == y_pred: return 1
    else: return 0

def quadratic_loss(y_real, y_pred):
    '''平方损失函数'''
    return (y_real - y_pred)**2

def absolute_loss(y_real, y_pred):
    '''绝对损失函数'''
    return abs(y_real, y_pred)

def huber_loss(y_real, y_pred, sigma=0.5):
    error = abs(y_real - y_pred)
    if error <= sigma:
        return 0.5*error*error
    else:
        return sigma*error - 0.5*sigma*sigma


def cross_entropy_loss(y, p):
    result = -y * math.log(p) - (1-y) * math.log(1-p)



if __name__ == '__main__':
   pass



