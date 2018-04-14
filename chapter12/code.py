#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W1401
"""
Created on Mon Nov  6 21:04:24 2017

@author: lu
"""

import numpy as np

import pandas as pd
from sqlalchemy import create_engine


"""
这部分代码主要是用Python连接数据库，提取数据进行分析。
所j以在运行代码之前需要讲sql语句运行一遍将数据插入到mysql数据库中
注意这里需要提前创建一个database，并且在开头增加使用database的语句
mysql -uroot -p < 7law.sql
需要等待一会

此部分代码没有运行，存在一定问题

count107-->统计107类别情况
programmer_1-->大概了解了处理数据意图
programmer_2-->提取所需数据，并且保存到数据库中
programmer_3-->进行数据筛选，保存到数据库中
programmer_4-->合并某些特征为一个特征，保存到数据库
programmer_5-->推荐矩阵
"""


def count107(i):
    j = i[["fullURL"]][i["fullURLId"].str.contains("107")].copy()
    # 添加空列
    j["type"] = None
    #  利用正则进行匹配，并重命名
    j["type"][j["fullURL"].str.contains("info/.+?/")] = u"知识首页"
    j["type"][j["fullURL"].str.contains("info/.+?/.+?")] = u"知识列表页"
    j["type"][j["fullURL"].str.contains("/\d+?_*\d+?\.html")] = u"知识内容页"
    return j["type"].value_counts()


def programmer_1():
    """
    用pymysql连接本地数据库
    按个人情况进行更改连接语句
    engine表示连接数据的引擎，chunksize表示每次读取数据量
    此时‘sql’只是一个容器
    """
    engine = create_engine(
        "mysql+pymysql://root:password@host:port/database_name?charset=utf8")
    sql = pd.read_sql("all_gzdata", engine, chunksize=10000)

    # 分别统计，并且合并相同项（按index分组求和）
    counts = [i["fullURLId"].value_counts() for i in sql]
    counts = pd.concat(counts).groupby(level=0).sum()
    # 自动重新设置index并将原来的index作为columns
    counts = counts.reset_index()
    counts.columns = ["index", "num"]
    # 修改列名，提取每个列名前三个数字，用到了正则表达式
    counts["type"] = counts["index"].str.extract("(\d{3})")
    counts_ = counts[["type", "num"]].groupby("type").sum()
    # 按类别排序
    counts_.sort_values("num", ascending=False)

    # 同counts1
    sql = pd.read_sql("all_gzdata", engine, chunksize=10000)
    counts2 = [count107(i) for i in sql]
    counts2 = pd.concat(counts2).groupby(level=0).sum()

    # 统计次数，同上分块统计结果并合并t
    c = [i["realIP"].value_counts() for i in sql]
    counts3 = pd.concat(counts2).groupby(level=0).sum()
    counts3 = pd.DataFrame(counts3)
    # 添加新列，全为1,统计某特征分别出现的次数
    counts3[1] = 1
    counts3.groupby(0).sum()
    return c

def programmer_2():
    engine = create_engine(
        "mysql+pymysql://root:password@host:port/database_name?charset=utf8")
    sql = pd.read_sql("sql_gzdata", engine, chunksize=10000)

    for i in sql:
        d = i[["realIP", "fullURL"]]
        d = d[d["fullURL"].str.contains("\.html")].copy()
        d.to_sql("cleaned_gzdata", engine, index=False, if_exists="append")


def programmer_3():
    engine = create_engine(
        "mysql_pymysql://root:password@host:port/database_name?charset=utf8")
    sql = pd.read_sql("cleaned_gzdata", engine, chunksize=10000)

    for i in sql:
        d = i.copy()
        # 替换关键词
        d["fullURL"] = d["fullURL"].str.replace("_\d{0,2}.html", ".html")
        # 去除重复数据
        d = d.drop_duplicates()
        d.to_sql("changed_gzdata", engine, index=False, if_exists="append")


def programmer_4():
    engine = create_engine(
        "mysql+pymysql://root:password@host:port/database_name?charset=utf8")
    sql = pd.read_sql("changed_gzdata", engine, chunksize=10000)

    for i in sql:
        d = i.copy()
        d["type_1"] = d["fullURL"]
        d["type_1"][d["fullURL"].str.contains("(ask)|(askzt)")] = "zixun"
        d.to_sql("splited_gzdata", engine, index=False, if_exists="append")


def Jaccard(a, b):
    return 1.0 * (a * b).sum() / (a + b - a * b).sum()


def programmer_5():
    class Recommender():

        sim = None

        # 判断距离（相似性）
        def similarity(self, x, distance):
            y = np.ones((len(x), len(x)))
            for i in range(len(x)):
                for j in range(len(x)):
                    y[i, j] = distance(x[i], x[j])

            return y

        def fit(self, x, distance=Jaccard):
            self.sim = self.similarity(x, distance)
        
        # 推荐矩阵
        def recommend(self, a):
            return np.dot(self.sim, a) * (1 - a)

    Recommender()

if __name__ == "__main__":
    programmer_1()
    programmer_2()
    programmer_3()
    programmer_4()
    programmer_5()
    pass
