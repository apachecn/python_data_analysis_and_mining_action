# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.cluster import KMeans
"""
programmer_1-->关于原始数据的一些特征描述并保存为新表，使用describe函数
programmer_2-->对原始数据进行清理，对其中某些数据做运算，并进行保存
programmer_3-->标准化数据并进行保存
programmer_4-->使用KMeans对数据进行聚类分析
"""


def programmer_1():

    datafile = 'data/air_data.csv'
    resultfile = 'tmp/explore.xls'

    data = pd.read_csv(datafile, encoding='utf-8')

    # 包括对数据的基本描述，percentiles参数是指定计算多少的分位数表（如1/4分位数、中位数等）；T是转置，转置后更方便查阅
    explore = data.describe(percentiles=[], include='all').T
    # describe()函数自动计算非空值数，需要手动计算空值数
    explore['null'] = len(data) - explore['count']

    explore = explore[['null', 'max', 'min']]
    explore.columns = [u'空值数', u'最大值', u'最小值']
    '''这里只选取部分探索结果。
    describe()函数自动计算的字段有count（非空值数）、unique（唯一值数）、top（频数最高者）、freq（最高频数）、mean（平均值）、std（方差）、min（最小值）、50%（中位数）、max（最大值）'''

    explore.to_excel(resultfile)


def programmer_2():
    datafile = 'data/air_data.csv'
    cleanedfile = 'tmp/data_cleaned.csv'

    data = pd.read_csv(datafile, encoding='utf-8')

    # 使用乘法运算非空数值的数据，因为numpy不支持*运算，在这里换做&运算
    data = data[data['SUM_YR_1'].notnull() & data['SUM_YR_2'].notnull()]

    # 只保留票价非零的，或者平均折扣率与总飞行公里数同时为0的记录。
    index1 = data['SUM_YR_1'] != 0
    index2 = data['SUM_YR_2'] != 0
    index3 = (data['SEG_KM_SUM'] == 0) & (data['avg_discount'] == 0)  # 该规则是“与”
    data = data[index1 | index2 | index3]  # 该规则是“或”

    data.to_csv(cleanedfile)


def programmer_3():

    datafile = 'data/zscoredata.xls'
    zscoredfile = 'tmp/zscoreddata.xls'

    data = pd.read_excel(datafile)
    # 核心语句，实现标准化变换，类似地可以实现任何想要的变换。
    data = (data - data.mean(axis=0)) / (data.std(axis=0))
    data.columns = ['Z' + i for i in data.columns]

    data.to_excel(zscoredfile, index=False)


def programmer_4():
    inputfile = 'tmp/zscoreddata.xls'
    k = 5
    data = pd.read_excel(inputfile)

    kmodel = KMeans(n_clusters=k, n_jobs=4)
    kmodel.fit(data)

    print(kmodel.cluster_centers_)  # 查看聚类中心
    print(kmodel.labels_)  # 查看各样本对应的类别


if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    # programmer_4()
    pass
