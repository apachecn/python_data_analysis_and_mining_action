#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 23:05:54 2017

@author: lu
"""

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from sklearn.linear_model import Lasso


"""
GM11-->自定义的灰度预测函
programmer_1-->读取文件提取基本信息
programmer_2-->用自定义的灰度预测函数，进行预测
programmer_3-->建立神经网络模型，进行预测并画图预测图
programmer_4-->使用自定义的灰度预测模型进行预测一组数据，并且画图
"""


def GM11(x0):
    # 1-AGO序列, 累计求和
    x1 = np.cumsum(x0)
    # 紧邻均值（ＭＥＡＮ）生成序列
    z1 = (x1[:-1] + x1[1:]) / 2.0
    z1 = z1.reshape(len(z1), 1)
    B = np.append(-z1, np.ones_like(z1), axis=1)
    Yn = x0[1:].reshape((len(x0) - 1, 1))
    # 矩阵计算，计算参数
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)
    # 还原值

    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))

    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) <
               0.6745 * x0.std()).sum() / len(x0)
    # 灰度预测函数、a、b、首项、方差比、小残差概率

    return f, a, b, x0[0], C, P


def programmer_1(inputfile, data_range):
    # inputfile = "data/data1.csv"
    data = pd.read_csv(inputfile)
    """
    原始方法，替代方法可以使用describe()方法，然后进行筛选
    r = [data.min(), data.max(), data.mean(), data.std()]
    r = pd.DataFrame(r, index = ["Min", "Max", "Mean", "STD"]).T
    """
    r = pd.DataFrame(data.describe()).T
    np.round(r, 2)

    # 计算相关系数矩阵
    np.round(data.corr(method="pearson"), 2)

    """
    原代码使用的是AdaptiveLasso，现更新为Lasso
    参数也由gamma变为tol（有待验证）
    """
    model = Lasso(tol=1)
    model.fit(data.iloc[:, 0:data_range], data["y"])
    # 各个特征的系数
    model.coef_
    print(model.coef_)


def programmer_2(inputfile, outputfile, startyear, feature_lst, roundnum=0):
    """
    year： 开始年份
    feature_lst: 特征列
    roundnum： 四舍五入保留的位数
    """
    data = pd.read_csv(inputfile)
    data.index = range(startyear, 2014)

    data.loc[2014] = None
    data.loc[2015] = None
    for i in feature_lst:
        f = GM11(data[i][list(range(startyear, 2014))].as_matrix())[0]
        # 2014年预测结果
        data[i][2014] = f(len(data) - 1)
        # 2015年预测结果
        data[i][2015] = f(len(data))
        data[i] = data[i].round(roundnum)

    print(data[feature_lst + ["y"]])
    data[feature_lst + ["y"]].to_excel(outputfile)


def programmer_3(inputfile, outputfile, modelfile, feature_lst, startyear, input_dim_1, units1, input_dim_2, units2, epochs_num=10000, roundnum=0):
    """
    feature_lst: 特征列
    input_dim、units: 表示训练模型层数和神经元个数
    roundnum: 四舍五入
    """

    data = pd.read_excel(inputfile)
    # 特征列
    # 取startyear年以前的数据
    data_train = data.loc[range(startyear, 2014)].copy()
    data_mean = data_train.mean()
    data_std = data.std()
    # 数据标准化
    data_train = (data_train - data_mean) / data_std
    # 特征数据
    x_train = data_train[feature_lst].as_matrix()
    # 标签数据
    y_train = data_train["y"].as_matrix()

    model = Sequential()
    model.add(Dense(input_dim=input_dim_1, units=units1))
    model.add(Activation("relu"))
    model.add(Dense(input_dim=input_dim_2, units=units2))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(x_train, y_train, epochs=epochs_num, batch_size=16)
    model.save_weights(modelfile)

    # 预测，并且还原结果
    x = ((data[feature_lst] - data_mean[feature_lst]) /
         data_std[feature_lst]).as_matrix()
    data["y_pred"] = model.predict(x) * data_std["y"] + data_mean["y"]
    data["y_pred"] = data["y_pred"].round(roundnum)

    data.to_excel(outputfile)
    # 画出预测结果图
    data[["y", "y_pred"]].plot(subplots=True, style=["b-o", "r-*"])
    plt.show()


def programmer_4():
    x0 = np.array([3152063, 2213050, 4050122,
                   5265142, 5556619, 4772843, 9463330])
    f, a, b, x00, C, P = GM11(x0)
    print(a, b, x00, C, P)
    print(u'2014年、2015年的预测结果分别为：\n%0.2f万元和%0.2f万元' % (f(8), f(9)))
    print(u'后验差比值为：%0.4f' % C)
    p = pd.DataFrame(x0, columns=["y"], index=range(2007, 2014))
    p.loc[2014] = None
    p.loc[2015] = None
    p["y_pred"] = [f(i) for i in range(1, 10)]
    p["y_pred"] = p["y_pred"].round(2)
    p.index = pd.to_datetime(p.index, format="%Y")

    p.plot(style=["b-o", "r-*"], xticks=p.index)
    plt.show()


if __name__ == "__main__":
    # programmer_1(inputfile="data/data1.csv",
    #              data_range=13)
    # programmer_2(inputfile="data/data1.csv",
    #              outputfile="tmp/data1_GM11.xls",
    #              startyear=1994,
    #              feature_lst=["x1", "x2", "x3", "x4", "x5", "x7"],
    #              roundnum=2)
    # programmer_3(inputfile="tmp/data1_GM11.xls",
    #              outputfile="data/revenue.xls",
    #              modelfile="tmp/1-net.model",
    #              feature_lst=["x1", "x2", "x3", "x4", "x5", "x7"],
    #              startyear=1994,
    #              input_dim_1=6,
    #              units1=12,
    #              input_dim_2=12,
    #              units2=1)

    # programmer_1(inputfile="data/data2.csv",
    #              data_range=6)
    # programmer_2(inputfile="data/data2.csv",
    #              outputfile="tmp/data2_GM11.xls",
    #              startyear=1999,
    #              feature_lst=["x1", "x3", "x5"],
    #              roundnum=6)
    # programmer_3(inputfile="tmp/data2_GM11.xls",
    #              outputfile="data/VAT.xls",
    #              modelfile="tmp/2-net.model",
    #              feature_lst=["x1", "x3", "x5"],
    #              startyear=1999,
    #              input_dim_1=3,
    #              units1=6,
    #              input_dim_2=6,
    #              units2=1,
    #              roundnum=2)

    # programmer_1(inputfile="data/data3.csv",
    #              data_range=10)
    # programmer_2(inputfile="data/data3.csv",
    #              outputfile="tmp/data3_GM11.xls",
    #              startyear=1999,
    #              feature_lst=["x3", "x4", "x6", "x8"])
    # programmer_3(inputfile="tmp/data3_GM11.xls",
    #              outputfile="data/sales_tax.xls",
    #              modelfile="tmp/3-net.model",
    #              feature_lst=["x3", "x4", "x6", "x8"],
    #              startyear=1999,
    #              input_dim_1=4,
    #              units1=8,
    #              input_dim_2=8,
    #              units2=1,
    #              roundnum=2)

    # programmer_1(inputfile="data/data4.csv",
    #              data_range=10)
    # programmer_2(inputfile="data/data4.csv",
    #              outputfile="tmp/data4_GM11.xls",
    #              startyear=2002,
    #              feature_lst=["x1", "x2", "x3", "x4", "x6", "x7", "x9", "x10"],
    #              roundnum=2)
    # programmer_3(inputfile="tmp/data4_GM11.xls",
    #              outputfile="data/enterprise_incomt.xls",
    #              modelfile="tmp/4-net.model",
    #              feature_lst=["x1", "x2", "x3", "x4", "x6", "x7", "x9", "x10"],
    #              startyear=2002,
    #              input_dim_1=8,
    #              units1=6,
    #              input_dim_2=6,
    #              units2=1,
    #              roundnum=2)

    # programmer_1(inputfile="data/data5.csv",
    #              data_range=7)
    # programmer_2(inputfile="data/data5.csv",
    #              outputfile="tmp/data5_GM11.xls",
    #              startyear=2000,
    #              feature_lst=["x1", "x4", "x5", "x7"])
    # programmer_3(inputfile="tmp/data5_GM11.xls",
    #              outputfile="data/personal_Income.xls",
    #              modelfile="tmp/5-net.model",
    #              feature_lst=["x1", "x4", "x5", "x7"],
    #              startyear=2000,
    #              input_dim_1=4,
    #              units1=8,
    #              input_dim_2=8,
    #              units2=1,
    #              epochs_num=15000# )

    # programmer_4()
    pass