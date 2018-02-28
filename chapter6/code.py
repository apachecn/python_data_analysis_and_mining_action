#-*- coding: utf-8 -*-
# pylint: disable=E1101
from random import shuffle

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from scipy.interpolate import lagrange
from sklearn.externals import joblib
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.tree import DecisionTreeClassifier
"""
cm_plot-->自定义混淆矩阵可视化
programmer_1-->使用拉格朗日插值法进行插值
programmer_2-->构建CART决策树模型，进行预测给出训练结果，并且绘制ROC曲线
programmer_3-->使用神经网络模型，进行预测给出训练结果，并且绘制ROC曲线
"""


def cm_plot(y, yp):
    cm = confusion_matrix(y, yp)

    plt.matshow(cm, cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(
                cm[x, y],
                xy=(x, y),
                horizontalalignment='center',
                verticalalignment='center')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def programmer_1():
    inputfile = 'data/missing_data.xls'
    outputfile = 'tmp/missing_data_processed.xls'

    data = pd.read_excel(inputfile, header=None)  # 读入数据

    # 自定义列向量插值函数
    # s为列向量，n为被插值的位置，k为取前后的数据个数，默认为5
    def ployinterp_column(s, n, k=5):
        y = s[list(range(n - k, n)) + list(range(n + 1, n + 1 + k))]  # 取数
        y = y[y.notnull()]  # 剔除空值
        return lagrange(y.index, list(y))(n)  # 插值并返回插值结果

    # 逐个元素判断是否需要插值
    for i in data.columns:
        for j in range(len(data)):
            if (data[i].isnull())[j]:
                data[i][j] = ployinterp_column(data[i], j)

    data.to_excel(outputfile, header=None, index=False)


def programmer_2():
    datafile = 'data/model.xls'
    data = pd.read_excel(datafile)
    data = data.as_matrix()
    shuffle(data)  # 随机打乱数据

    # 设置训练数据比8:2
    p = 0.8
    train = data[:int(len(data) * p), :]
    test = data[int(len(data) * p):, :]

    # 构建CART决策树模型
    treefile = 'tmp/tree.pkl'
    tree = DecisionTreeClassifier()
    tree.fit(train[:, :3], train[:, 3])

    joblib.dump(tree, treefile)

    cm_plot(train[:, 3], tree.predict(train[:, :3])).show()  # 显示混淆矩阵可视化结果
    # 注意到Scikit-Learn使用predict方法直接给出预测结果。

    fpr, tpr, thresholds = roc_curve(
        test[:, 3], tree.predict_proba(test[:, :3])[:, 1], pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label='ROC of CART', color='green')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # 设定边界范围
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc=4)
    plt.show()
    print(thresholds)


def programmer_3():
    datafile = 'data/model.xls'
    data = pd.read_excel(datafile)
    data = data.as_matrix()
    shuffle(data)

    p = 0.8
    train = data[:int(len(data) * p), :]
    test = data[int(len(data) * p):, :]

    # 构建LM神经网络模型
    netfile = 'tmp/net.model'

    net = Sequential()  # 建立神经网络
    #    net.add(Dense(input_dim = 3, units = 10))
    # 添加输入层（3节点）到隐藏层（10节点）的连接
    net.add(Dense(10, input_shape=(3, )))
    net.add(Activation('relu'))  # 隐藏层使用relu激活函数
    #    net.add(Dense(input_dim = 10, units = 1))
    #添加隐藏层（10节点）到输出层（1节点）的连接
    net.add(Dense(1, input_shape=(10, )))
    net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
    net.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        sample_weight_mode="binary")  # 编译模型，使用adam方法求解

    net.fit(train[:, :3], train[:, 3], epochs=100, batch_size=1)
    net.save_weights(netfile)

    predict_result = net.predict_classes(train[:, :3]).reshape(
        len(train))  # 预测结果变形
    '''这里要提醒的是，keras用predict给出预测概率，predict_classes才是给出预测类别，而且两者的预测结果都是n x 1维数组，而不是通常的 1 x n'''

    cm_plot(train[:, 3], predict_result).show()

    predict_result = net.predict(test[:, :3]).reshape(len(test))
    fpr, tpr, thresholds = roc_curve(test[:, 3], predict_result, pos_label=1)
    plt.plot(fpr, tpr, linewidth=2, label='ROC of LM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.ylim(0, 1.05)
    plt.xlim(0, 1.05)
    plt.legend(loc=4)
    plt.show()
    print(thresholds)


if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    pass