# -*- coding: utf-8 -*-
# pylint: disable=E1101
"""
Created on Fri Oct 20 16:06:09 2017

@author: wnma3
"""

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers.core import Activation, Dense
from keras.models import Sequential
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import export_graphviz
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
"""
programmer_1-->使用随机森林算出有效特征，使用线性回归计算相关系数
programmer_2-->使用决策数模型，生成决策树过程并保存为dot文件，天气、周末、促销决定销量
programmer_3-->使用Keras神经网络模型，训练数据预测销量高低
cm_plot-->自定义混淆矩阵可视化
density_plot-->自定义概率密度图函数
programmer_4-->使用KMeans聚类，做可视化操作（概率密度图）
programmer_5-->继programmer_4将数据做降维处理，并且可视化不同聚类的类别
programmer_6-->进行白噪声、平稳性检测，建立ARIMA(0, 1, 1)模型预测之后五天的结果
programmer_7-->使用Kmeans聚类之后，画出散点图，标记离群点
find_rule-->寻找关联规则的函数
connect_string-->自定义连接函数，用于实现L_{k-1}到C_k的连接
programmer_8-->菜单中各个菜品的关联程度
"""


def programmer_1():
    filename = "data/bankloan.xls"
    data = pd.read_excel(filename)

    x = data.iloc[:, :8].as_matrix()
    y = data.iloc[:, 8].as_matrix()

    rlr = RLR()
    rlr.fit(x, y)
    rlr_support = rlr.get_support()
    support_col = data.drop('违约', axis=1).columns[rlr_support]

    print(
        "rlr_support_columns: {columns}".format(columns=','.join(support_col)))
    x = data[support_col].as_matrix()

    lr = LR()
    lr.fit(x, y)

    print("lr: {score}".format(score=lr.score(x, y)))


def programmer_2():
    inputfile = "data/sales_data.xls"
    data = pd.read_excel(inputfile, index_col=u'序号')

    data[data == u'是'] = 1
    data[data == u'高'] = 1
    data[data == u'好'] = 1
    data[data != 1] = -1

    x = data.iloc[:, :3].as_matrix().astype(int)
    y = data.iloc[:, 3].as_matrix().astype(int)

    dtc = DTC()
    dtc.fit(x, y)

    x = pd.DataFrame(x)
    with open("tree.dot", "w") as f:
        f = export_graphviz(dtc, feature_names=x.columns, out_file=f)


def programmer_3():
    inputfile = "data/sales_data.xls"
    data = pd.read_excel(inputfile, index_col=u'序号')

    data[data == u'好'] = 1
    data[data == u'是'] = 1
    data[data == u'高'] = 1
    data[data != 1] = 0

    x = data.iloc[:, :3].as_matrix().astype(int)
    y = data.iloc[:, 3].as_matrix().astype(int)

    # 进行神经网络训练
    model = Sequential()
    model.add(Dense(input_dim=3, units=10))
    model.add(Activation('relu'))
    model.add(Dense(input_dim=10, units=1))
    model.add(Activation('sigmoid'))
    # 修正神经网络
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(x, y, epochs=1000, batch_size=10)

    yp = model.predict_classes(x).reshape(len(y))

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

    cm_plot(y, yp).show()


def programmer_4():
    inputfile = 'data/consumption_data.xls'
    outputfile = 'tmp/data_type.xls'
    """
    k: 聚类类别
    iteration: 聚类循环次数
    model.labels_： 聚类类别
    model.cluster_centers_： 聚类中心
    """
    k = 3
    iteration = 500
    data = pd.read_excel(inputfile, index_col='Id')
    data_zs = 1.0 * (data - data.mean()) / data.std()

    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    model.fit(data_zs)

    # 统计各个类别的数目
    r1 = pd.Series(model.labels_).value_counts()
    r2 = pd.DataFrame(model.cluster_centers_)
    r = pd.concat([r2, r1], axis=1)
    r.columns = list(data.columns) + [u'类别数目']
    print(r)

    # 详细输出每个样本对应的类别
    r = pd.concat([data, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'聚类类别']
    r.to_excel(outputfile)

    def density_plot(data, k):
        p = data.plot(kind='kde', linewidth=2, subplots=True, sharex=False)
        [p[i].set_ylabel(u'密度') for i in range(k)]
        plt.legend()
        return plt

    # 保存概率密度图
    pic_output = 'tmp/pd_'
    for i in range(k):
        density_plot(data[r[u'聚类类别'] == i],
                     k).savefig(u'%s%s.png' % (pic_output, i))

    return data_zs, r


def programmer_5(data_zs, r):
    # 进行数据降维
    tsne = TSNE()
    tsne.fit_transform(data_zs)
    tsne = pd.DataFrame(tsne.embedding_, index=data_zs.index)

    # 不同类别用不同颜色和样式绘图
    d = tsne[r[u'聚类类别'] == 0]
    plt.plot(d[0], d[1], 'r.')
    d = tsne[r[u'聚类类别'] == 1]
    plt.plot(d[0], d[1], 'go')
    d = tsne[r[u'聚类类别'] == 2]
    plt.plot(d[0], d[1], 'b*')
    plt.show()


def programmer_6():
    """
    警告解释：
    # UserWarning: matplotlib is currently using a non-GUI backend, so cannot show the figure
  "matplotlib is currently using a non-GUI backend, "
    调用了多次plt.show()
    解决方案，使用plt.subplot()

    # RuntimeWarning: overflow encountered in exp
    运算精度不够

    forecastnum-->预测天数
    plot_acf().show()-->自相关图
    plot_pacf().show()-->偏自相关图
    """
    discfile = 'data/arima_data.xls'
    forecastnum = 5
    data = pd.read_excel(discfile, index_col=u'日期')

    fig = plt.figure(figsize=(8, 6))
    # 第一幅自相关图
    ax1 = plt.subplot(411)
    fig = plot_acf(data, ax=ax1)

    # 平稳性检测
    print(u'原始序列的ADF检验结果为：', ADF(data[u'销量']))
    # 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

    # 差分后的结果
    D_data = data.diff().dropna()
    D_data.columns = [u'销量差分']
    # 时序图
    D_data.plot()
    plt.show()
    # 第二幅自相关图
    fig = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(412)
    fig = plot_acf(D_data, ax=ax2)
    # 偏自相关图
    ax3 = plt.subplot(414)
    fig = plot_pacf(D_data, ax=ax3)
    plt.show()
    fig.clf()

    print(u'差分序列的ADF检验结果为：', ADF(D_data[u'销量差分']))  # 平稳性检测

    # 白噪声检验
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值
    data[u'销量'] = data[u'销量'].astype(float)
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # bic矩阵
    data.dropna(inplace=True)

    # 存在部分报错，所以用try来跳过报错；存在warning，暂未解决使用warnings跳过
    import warnings
    warnings.filterwarnings('error')
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    # 用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
    model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
    model.summary2()  # 给出一份模型报告
    model.forecast(forecastnum)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。


def programmer_7():
    """
    k：聚类中心数
    threshold：离散点阈值
    iteration：聚类最大循环次数
    """
    inputfile = 'data/consumption_data.xls'
    k = 3
    threshold = 2
    iteration = 500
    data = pd.read_excel(inputfile, index_col='Id')
    # 数据标准化
    data_zs = 1.0 * (data - data.mean()) / data.std()

    model = KMeans(n_clusters=k, n_jobs=4, max_iter=iteration)
    model.fit(data_zs)

    # 标准化数据及其类别
    # 每个样本对应的类别
    r = pd.concat(
        [data_zs, pd.Series(model.labels_, index=data.index)], axis=1)
    r.columns = list(data.columns) + [u'聚类类别']

    norm = []
    for i in range(k):  # 逐一处理
        norm_tmp = r[['R', 'F', 'M']][r[u'聚类类别'] == i] - \
            model.cluster_centers_[i]
        # 求出绝对距离
        norm_tmp = norm_tmp.apply(np.linalg.norm, axis=1)
        # 求相对距离并添加
        norm.append(norm_tmp / norm_tmp.median())

    norm = pd.concat(norm)
    # 正常点
    norm[norm <= threshold].plot(style='go')
    # 离群点
    discrete_points = norm[norm > threshold]
    discrete_points.plot(style='ro')
    # 标记离群点
    for i in range(len(discrete_points)):
        _id = discrete_points.index[i]
        n = discrete_points.iloc[i]
        plt.annotate('(%s, %0.2f)' % (_id, n), xy=(_id, n), xytext=(_id, n))

    plt.xlabel(u'编号')
    plt.ylabel(u'相对距离')
    plt.show()


def connect_string(x, ms):
    x = list(map(lambda i: sorted(i.split(ms)), x))
    l = len(x[0])
    r = []
    for i in range(len(x)):
        for j in range(i, len(x)):
            if x[i][:l - 1] == x[j][:l - 1] and x[i][l - 1] != x[j][l - 1]:
                r.append(x[i][:l - 1] + sorted([x[j][l - 1], x[i][l - 1]]))
    return r


def find_rule(d, support, confidence, ms=u'--'):
    # 定义输出结果
    result = pd.DataFrame(index=['support', 'confidence'])
    # 支持度序列
    support_series = 1.0 * d.sum() / len(d)
    # 支持度第一次筛选
    column = list(support_series[support_series > support].index)
    k = 0

    while len(column) > 1:
        k = k + 1
        print(u'\n正在进行第%s次搜索...' % k)
        column = connect_string(column, ms)
        print(u'数目：%s...' % len(column))

        # 新一批支持度的计算函数
        def sf(i):
            return d[i].prod(axis=1, numeric_only=True)

        # 创建连接数据，这一步耗时、耗内存最严重。当数据集较大时，可以考虑并行运算优化。
        d_2 = pd.DataFrame(
            list(map(sf, column)), index=[ms.join(i) for i in column]).T

        # 计算连接后的支持度
        support_series_2 = 1.0 * \
            d_2[[ms.join(i) for i in column]].sum() / len(d)
        column = list(
            support_series_2[support_series_2 > support].index)  # 新一轮支持度筛选
        support_series = support_series.append(support_series_2)
        column2 = []

        for i in column:  # 遍历可能的推理，如{A,B,C}究竟是A+B-->C还是B+C-->A还是C+A-->B？
            i = i.split(ms)
            for j in range(len(i)):
                column2.append(i[:j] + i[j + 1:] + i[j:j + 1])

        cofidence_series = pd.Series(
            index=[ms.join(i) for i in column2])  # 定义置信度序列
        # 计算置信度序列
        for i in column2:
            cofidence_series[ms.join(i)] = support_series[ms.join(
                sorted(i))] / support_series[ms.join(i[:len(i) - 1])]
        # 置信度筛选
        for i in cofidence_series[cofidence_series > confidence].index:
            result[i] = 0.0
            result[i]['confidence'] = cofidence_series[i]
            result[i]['support'] = support_series[ms.join(sorted(i.split(ms)))]

    result = result.T.sort_values(
        ['confidence', 'support'], ascending=False)  # 结果整理，输出
    print(u'\n结果为：')
    print(result)

    return result


def programmer_8():
    inputfile = 'data/menu_orders.xls'
    outputfile = 'tmp/apriori_rules.xls'
    data = pd.read_excel(inputfile, header=None)

    print(u'\n转换原始数据至0-1矩阵...')

    # 转换0-1矩阵的过渡函数
    def ct(x):
        return pd.Series(1, index=x[pd.notnull(x)])

    b = map(ct, data.as_matrix())
    data = pd.DataFrame(list(b)).fillna(0)
    print(u'\n转换完毕。')
    del b  # 删除中间变量b，节省内存

    support = 0.2  # 最小支持度
    confidence = 0.5  # 最小置信度
    ms = '---'  # 连接符，默认'--'，用来区分不同元素，如A--B。需要保证原始表格中不含有该字符

    find_rule(data, support, confidence, ms).to_excel(outputfile)


if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    # data_zs, r = programmer_4()
    # programmer_5(data_zs, r)
    # programmer_6()
    # programmer_7()
    # programmer_8()
    pass
