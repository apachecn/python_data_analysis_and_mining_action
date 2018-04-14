#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Created on Sat Nov  4 11:04:32 2017

@author: lu
"""

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF


"""
FutureWarning警告：原因未知，在spyder3上运行第二次就消失了，猜测是使用了缓存的原因
attr_trans-->属性变换
programmer_1-->数据筛选
programmer_2-->平稳性检测
programmer_3-->白噪声检测
programmer_4-->确定最佳p、d、q值，有问题！！！
programmer_5-->模型检验
programmer_6-->计算预测误差
"""


# 属性变换,改变列名
def attr_trans(x):
    result = pd.Series(
        index=["SYS_NAME", "CWXT_DB:184:C:\\", "CWXT_DB:184:D:\\", "COLLECTTIME"])
    result["SYS_NAME"] = x["SYS_NAME"].iloc[0]
    result["COLLECTTIME"] = x["COLLECTTIME"].iloc[0]
    result["CWXT_DB:184:C:\\"] = x["VALUE"].iloc[0]
    result["CWXT_DB:184:D:\\"] = x["VALUE"].iloc[1]

    return result


def programmer_1():

    discfile = "data/discdata.xls"
    transformeddata = "tmp/discdata_processed.xls"

    data = pd.read_excel(discfile)
    # 提取某部分数据
    data = data[data["TARGET_ID"] == 184].copy()
    # 以某字段进行分组
    data_group = data.groupby("COLLECTTIME")

    # 逐组处理
    data_processed = data_group.apply(attr_trans)
    data_processed.to_excel(transformeddata, index=False)


def programmer_2():
    discfile = "data/discdata_processed.xls"
    data = pd.read_excel(discfile)
    # 去除最后5个数据
    predictnum = 5
    data = data.iloc[:len(data) - predictnum]

    # 平稳性检测
    diff = 0
    adf = ADF(data["CWXT_DB:184:D:\\"])
    while adf[1] > 0.05:
        diff = diff + 1
        adf = ADF(data["CWXT_DB:184:D:\\"].diff(diff).dropna())

    print(u"原始序列经过%s阶差分后归于平稳，p值为%s" % (diff, adf[1]))


def programmer_3():

    discfile = "data/discdata_processed.xls"

    data = pd.read_excel(discfile)
    data = data.iloc[:len(data) - 5]

    [[lb], [p]] = acorr_ljungbox(data["CWXT_DB:184:D:\\"], lags=1)
    if p < 0.05:
        print(u"原始序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"原始序列为白噪声序列，对应的p值为：%s" % p)

    [[lb], [p]] = acorr_ljungbox(
        data["CWXT_DB:184:D:\\"].diff().dropna(), lags=1)

    if p < 0.05:
        print(u"一阶差分序列为非白噪声序列，对应的p值为：%s" % p)
    else:
        print(u"一阶差分序列为白噪声序列，对应的p值为：%s" % p)
    print(lb)

def programmer_4():
    discfile = "data/discdata_processed.xls"

    data = pd.read_excel(discfile, index_col="COLLECTTIME")
    # 不使用最后五个数据
    data = data.iloc[:len(data) - 5]
    xdata = data["CWXT_DB:184:D:\\"]

    # 定阶
    pmax = int(len(xdata) / 10)
    qmax = int(len(xdata) / 10)
    # 定义bic矩阵
    bic_matrix = []
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(xdata, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)

    bic_matrix = pd.DataFrame(bic_matrix)
    # 找出最小值
    p, q = bic_matrix.stack().idxmin()
    print(u"BIC最小的p值和q值为：%s、%s" % (p, q))


def programmer_5():
    discfile = "data/discdata_processed.xls"
    # 残差延迟个数
    lagnum = 12

    data = pd.read_excel(discfile, index_col="COLLECTTIME")
    data = data.iloc[:len(data) - 5]
    xdata = data["CWXT_DB:184:D:\\"]

    # 训练模型并预测，计算残差
    arima = ARIMA(xdata, (0, 1, 1)).fit()
    xdata_pred = arima.predict(typ="levels")
    pred_error = (xdata_pred - xdata).dropna()

    lb, p = acorr_ljungbox(pred_error, lags=lagnum)
    h = (p < 0.05).sum()
    if h > 0:
        print(u"模型ARIMA（0,1,1)不符合白噪声检验")
    else:
        print(u"模型ARIMA（0,1,1)符合白噪声检验")
    print(lb)

def programmer_6():

    file = "data/predictdata.xls"
    data = pd.read_excel(file)

    # 计算误差
    abs_ = (data[u"预测值"] - data[u"实际值"]).abs()
    mae_ = abs_.mean()
    rmse_ = ((abs_ ** 2).mean()) ** 0.5
    mape_ = (abs_/data[u"实际值"]).mean()

    print(u"平均绝对误差为：%0.4f, \n 均方根误差为%0.4f, \n平均绝对百分误差为：%0.6f。" % (mae_, rmse_, mape_))



if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    # programmer_4()
    # programmer_5()
    # programmer_6()
    pass