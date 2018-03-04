#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 13:09:18 2017

@author: lu
"""

import numpy as np

import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
"""
programmer_1-->简单的数据筛选，划分数据
programmer_2-->阈值寻优？？？不懂。。
programmer_3-->建立训练神经网络，并进行模型的检验
programmer_4-->根据几个特征推算出是否满足某项条件
event_num-->相邻时间的差分，比较是否大于阈值
"""


def programmer_1():
    # 阈值
    threshold = pd.Timedelta("4 min")
    inputfile = "data/water_heater.xls"
    outputfile = "tmp/dividsequence.xls"

    data = pd.read_excel(inputfile)
    # dataframe处理
    data[u"发生时间"] = pd.to_datetime(data[u"发生时间"], format="%Y%m%d%H%M%S")
    data = data[data[u"水流量"] > 0]  # 流量大于0
    d = data[u"发生时间"].diff() > threshold  # 相邻时间作差分，大于threshold
    data[u"事件编号"] = d.cumsum() + 1  # 通过累积求和的方式为事件编号

    data.to_excel(outputfile)


# 相邻时间作差分，比较是否大于阈值


def programmer_2():
    inputfile = "data/water_heater.xls"
    # 使用之后四个点的平均斜率
    n = 4

    # 专家阈值
    threshold = pd.Timedelta(minutes=5)
    data = pd.read_excel(inputfile)
    data[u"发生时间"] = pd.to_datetime(data[u"发生时间"], format="%Y%m%d%H%M%S")
    data = data[data[u"水流量"] > 0]

    # 定义阈值列
    dt = [pd.Timedelta(minutes=i) for i in np.arange(1, 9, 0.25)]
    h = pd.DataFrame(dt, columns=[u"阈值"])

    def event_num(ts):
        d = data[u"发生时间"].diff() > ts
        # 返回事件数
        return d.sum() + 1

    # 计算每个阈值对应的事件数
    h[u"事件数"] = h[u"阈值"].apply(event_num)
    # 计算每两个相邻点对应的斜率
    h[u"斜率"] = h[u"事件数"].diff() / 0.25
    # 采用后n个的斜率绝对值平均作为斜率指标
    h[u"斜率指标"] = pd.Series.rolling(h[u"斜率"].abs(), n).mean()
    ts = h[u"阈值"][h[u"斜率指标"].idxmin() - n]

    if ts > threshold:
        ts = pd.Timedelta(minutes=4)

    print(ts)


def programmer_3():

    inputfile1 = "data/train_neural_network_data.xls"
    inputfile2 = "data/test_neural_network_data.xls"
    testoutputfile = "tmp/test_output_data.xls"

    # 读取训练集和测试集，并且划分样本特征和标签
    data_train = pd.read_excel(inputfile1)
    data_test = pd.read_excel(inputfile2)
    y_train = data_train.iloc[:, 4].as_matrix()
    x_train = data_train.iloc[:, 5:17].as_matrix()
    y_test = data_test.iloc[:, 4].as_matrix()
    x_test = data_test.iloc[:, 5:17].as_matrix()

    # 建立神经网络模型
    model = Sequential()
    model.add(Dense(17, input_shape=(11, )))
    model.add(Activation("relu"))
    model.add(Dense(10, input_shape=(17, )))
    model.add(Activation("relu"))
    model.add(Dense(1, input_shape=(10, )))
    model.add(Activation("sigmoid"))
    # 编译模型
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        sample_weight_mode="binary")
    # 训练模型
    model.fit(x_train, y_train, nb_epoch=100, batch_size=1)
    # 保存模型
    model.save_weights("tmp/net.model")

    # 进行预测
    r = pd.DataFrame(model.predict_classes(x_test), columns=[u"预测结果"])
    pd.concat([data_test.iloc[:, :5], r], axis=1).to_excel(testoutputfile)
    model.predict(x_test)
    return y_test


def programmer_4():
    threshold = pd.Timedelta("4 min")
    inputfile = "data/water_heater.xls"
    outputfile = "tmp/attribute_extract.xls"
    data = pd.read_excel(inputfile)

    data[u"发生时间"] = pd.to_datetime(data[u"发生时间"], format="%Y%m%d%H%M%S")
    data = data[data[u"水流量"] > 0]
    d = data[u"发生时间"].diff() > threshold
    data[u"事件编号"] = d.cumsum() + 1

    data_g = data.groupby(u"事件编号")
    result = pd.DataFrame()
    dt = pd.Timedelta(seconds=2)

    for _, g in data_g:
        temp = pd.DataFrame(index=[0])
        # 根据用水时长、开关机切换次数、总用水量推出是否是洗澡
        tstart = g[u"发生时间"].min()
        tend = g[u"发生时间"].max()
        temp[u"用水事件时长（M）"] = (dt + tend - tstart).total_seconds() / 60
        temp[u"开关机切换次数"] = (pd.Series.rolling(g[u"开关机状态"] == u"关",
                                              2).sum() == 1).sum()
        temp[u"总用水量（L）"] = g[u"水流量"].sum()
        tdiff = g[u"发生时间"].diff()
        if len(g[u"发生时间"]) == 1:
            temp[u"总用水时长（Min）"] = dt.total_seconds() / 60
        else:
            temp[u"总用水时长（Min）"] = (
                tdiff.sum() - tdiff.iloc[1] / 2 -
                tdiff.iloc[len(tdiff) - 1] / 2).total_seconds() / 60
        temp[u"平均水流量（L/min）"] = temp[u"总用水量（L）"] / temp[u"总用水时长（Min）"]
        result = result.append(temp, ignore_index=True)

    result.to_excel(outputfile)


if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    # programmer_4()
    pass