#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:10:35 2017

@author: lu
"""

import pickle

from numpy.random import shuffle

import pandas as pd
from sklearn import metrics, svm
"""
programmer_1-->svm支持向量机
"""


def programmer_1():

    inputfile = "data/moment.csv"
    outputfile1 = "tmp/cm_train.xls"
    outputfile2 = "tmp/cm_test.xls"
    data = pd.read_csv(inputfile, encoding="gbk")
    data = data.as_matrix()

    # 随即抽取训练集和验证集--8:2
    shuffle(data)
    data_train = data[:int(0.8 * len(data)), :]
    data_test = data[int(0.8 * len(data)):, :]
    # 训练集/验证集的训练数据和结果数据的抽取
    x_train = data_train[:, 2:] * 30
    y_train = data_train[:, 0].astype(int)
    x_test = data_test[:, 2:] * 30
    y_test = data_test[:, 0].astype(int)
    # 训练支持向量机的SVC
    model = svm.SVC()
    model.fit(x_train, y_train)

    # 保存模型
    pickle.dump(model, open("tmp/svm.model", "wb"))
    # 混淆矩阵，评估模型的准确性
    cm_train = metrics.confusion_matrix(y_train, model.predict(x_train))
    cm_test = metrics.confusion_matrix(y_test, model.predict(x_test))
    print(cm_train, '\n', cm_test)
    pd.DataFrame(
        cm_train, index=list(range(1, 6)),
        columns=list(range(1, 6))).to_excel(outputfile1)
    pd.DataFrame(
        cm_test, index=list(range(1, 6)),
        columns=list(range(1, 6))).to_excel(outputfile2)


if __name__ == "__main__":
    programmer_1()
