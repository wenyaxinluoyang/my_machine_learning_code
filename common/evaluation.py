import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def confusion_matrix(y_real, y_pred):
    '''计算混淆矩阵'''
    # True Positive, True Negative, False Positive, False Negative
    TP, TN, FP, FN = 0, 0, 0, 0
    for real, pred in zip(y_real, y_pred):
        if real == 1 and pred == 1: TP = TP + 1
        if real == 0 and pred == 0: TN = TN + 1
        if real == 1 and pred == 0: FN = FN + 1 # 假阴性
        if real == 0 and pred == 1: FP = FP + 1 # 假阳性
    return TP, TN, FP, FN


def split(y_real, y_pred):
    threshold_list = np.linspace(0, 1, num=50)
    df = pd.DataFrame()
    tp_list = []
    tn_list = []
    fp_list = []
    fn_list = []
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    tpr_list = []
    fpr_list = []
    for threshold in threshold_list:
        temp = np.where(y_pred > threshold, 1, 0)
        tp, tn, fp, fn = confusion_matrix(y_real, temp)
        tp_list.append(tp)
        tn_list.append(tn)
        fp_list.append(fp)
        fp_list.append(fn)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        accuracy_list.append(accuracy)
        precision = tp / (tp + fp)
        precision_list.append(precision)
        recall = tp / (tp + fn)
        recall_list.append(recall)
        f1_score_list.append(2*precision*recall/(precision + recall))
        tpr_list.append(recall)
        fpr_list.append(fp / (tn + fp))
    df['split_value'] = threshold_list
    df['True_Positive'] = tp_list
    df['True_Negative'] = tn_list
    df['False_Positive'] = fp_list
    df['False_Negative'] = fn_list
    df['Accuracy'] = accuracy_list
    df['Precision'] = precision_list
    df['Recall'] = recall_list
    df['F1_Score'] = f1_score_list
    df['TPR'] = tpr_list
    df['FPR'] = fpr_list
    return df

def plot_mode_roc(y_real, y_prob):
    pass

def plot_model_ks(y_real, y_prob):
    '''
    绘制模型的KS曲线
    :param y_real: 实际值， 要求值为 ndarray
    :param y_prob: 预测值, 要求值为 ndarray
    :return: KS曲线图
    '''
    total_bad = np.sum(y_real)
    total_good = len(y_real) - total_bad
    threshold_list = np.linspace(y_prob.min(), y_prob.max(), 200)
    good_rate = []
    bad_rate = []
    ks_list = []
    for threshold in threshold_list:
        index = np.argwhere(y_prob < threshold)
        labels = y_real[index]
        bad = np.sum(labels)
        good = len(labels) - bad
        goodrate = good / total_good
        badrate = bad / total_bad
        good_rate.append(goodrate)
        bad_rate.append(badrate)
        ks = abs(goodrate - badrate)
        ks_list.append(ks)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1,1,1)
    ax.plot(threshold_list, good_rate, color='green', label='good_rate')
    ax.plot(threshold_list, bad_rate, color='red', label='bad_rate')
    ax.plot(threshold_list, ks_list, color='blue', label='good-bad')
    index = np.argmax(np.array(ks_list))
    best_threshold = threshold_list[index]
    print(best_threshold, bad_rate[index], good_rate[index])
    ax.plot([best_threshold, best_threshold], [0, good_rate[index]], color='black', ls='--', alpha=0.5, label=f'best split point threshold={round(best_threshold, 2)}')
    ax.text(best_threshold+0.01, bad_rate[index]+0.01, str(round(bad_rate[index], 2)), color='red')
    ax.text(best_threshold-0.04, good_rate[index]+0.01, str(round(good_rate[index], 2)), color='green')
    ax.set_title('KS: {:.3f}'.format(ks_list[index]))
    ax.set_xlabel('split threshold')
    ax.set_ylabel('bad or good rate')
    ax.legend(loc='best')
    return plt.show()

def test_plot_model_ks():
    df = pd.read_csv('datasets/result.csv')
    # print(df['y_real'].value_counts())
    y_real = df['y_real'].values
    y_pred = df['y_pred'].values
    # plot_model_ks(y_real, y_pred)

if __name__ == '__main__':
    test_plot_model_ks()
    # split(None, None)
