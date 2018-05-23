#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#pylint: disable=W1401
"""
Created on Thu Nov  9 15:12:30 2017

@author: lu
"""

import jieba
import pandas as pd
from gensim import corpora, models


"""
由于每个阶段的数据文件存在依赖关系，所以这里输出保存在了data/目录下
programmer_1-->提取数据
programmer_2-->数据去重
programmer_3-->利用正则去除一些数据
programmer_4-->使用jieba分词
programmer_5-->分词之后的语义分析，LDA模型分析正面负面情感
"""


def programmer_1():
    inputfile = "data/huizong.csv"
    outputfile = "data/meidi_jd.txt"
    data = pd.read_csv(inputfile, encoding="utf-8")
    data = data[[u"评论"]][data[u"品牌"] == u"美的"]
    data.to_csv(outputfile, index=False, header=False, encoding="utf8")


def programmer_2():
    inputfile = "data/meidi_jd.txt"
    outputfile = "data/meidi_jd_process_1.txt"
    data = pd.read_csv(inputfile, encoding="utf8", header=None)
    l1 = len(data)
    data = pd.DataFrame(data[0].unique())
    l2 = len(data)
    data.to_csv(outputfile, index=False, header=False, encoding="utf8")
    print(u"删除了%s条评论" % (l1 - l2))


def programmer_3():
    inputfile1 = u"data/meidi_jd_process_end_负面情感结果.txt"
    inputfile2 = u"data/meidi_jd_process_end_正面情感结果.txt"
    outputfile1 = "data/meidi_jd_neg.txt"
    outputfile2 = "data/meidi_jd_pos.txt"

    data1 = pd.read_csv(inputfile1, encoding="utf8", header=None)
    data2 = pd.read_csv(inputfile2, encoding="utf8", header=None)

    data1 = pd.DataFrame(data1[0].str.replace(".*?\d+?\\t ", ""))
    data2 = pd.DataFrame(data2[0].str.replace(".*?\d+?\\t ", ""))

    data1.to_csv(outputfile1, index=False, header=False, encoding="utf8")
    data2.to_csv(outputfile2, index=False, header=False, encoding="utf8")


def programmer_4():

    inputfile1 = "data/meidi_jd_neg.txt"
    inputfile2 = "data/meidi_jd_pos.txt"
    outputfile1 = "data/meidi_jd_neg_cut.txt"
    outputfile2 = "data/meidi_jd_pos_cut.txt"

    data1 = pd.read_csv(inputfile1, encoding="utf8", header=None)
    data2 = pd.read_csv(inputfile2, encoding="utf8", header=None)

    def mycut(s): return " ".join(jieba.cut(s))

    data1 = data1[0].apply(mycut)
    data2 = data2[0].apply(mycut)

    data1.to_csv(outputfile1, index=False, header=False, encoding="utf8")
    data2.to_csv(outputfile2, index=False, header=False, encoding="utf8")

def programmer_5():
    negfile = "data/meidi_jd_neg_cut.txt"
    posfile = "data/meidi_jd_pos_cut.txt"
    stoplist = "data/stoplist.txt"

    neg = pd.read_csv(negfile, encoding="utf8", header=None)
    pos = pd.read_csv(posfile, encoding="utf8", header=None)
    """
    sep设置分割词，由于csv默认半角逗号为分割词，而且该词恰好位于停用词表中
    所以会导致读取错误
    解决办法是手动设置一个不存在的分割词，这里使用的是tipdm
    参数engine加上，指定引擎，避免警告
    """
    stop = pd.read_csv(stoplist, encoding="utf8", header=None, sep="tipdm", engine="python")

    # pandas自动过滤了空格，这里手动添加
    stop = [" ", ""] + list(stop[0])

    # 定义分割函数，然后用apply进行广播
    neg[1] = neg[0].apply(lambda s: s.split(" "))
    neg[2] = neg[1].apply(lambda x: [i for i in x if i not in stop])
    pos[1] = pos[0].apply(lambda s: s.split(" "))
    pos[2] = pos[1].apply(lambda x: [i for i in x if i not in stop])

    # 负面主题分析
    # 建立词典
    neg_dict = corpora.Dictionary(neg[2])
    # 建立语料库
    neg_corpus = [neg_dict.doc2bow(i) for i in neg[2]]
    # LDA模型训练
    neg_lda = models.LdaModel(neg_corpus, num_topics=3, id2word=neg_dict)

    for i in range(3):
        print(neg_lda.print_topic(i))

    # 正面主题分析
    # 以下同上
    pos_dict = corpora.Dictionary(pos[2])
    pos_corpus = [pos_dict.doc2bow(i) for i in pos[2]]
    pos_lda = models.LdaModel(pos_corpus, num_topics=3, id2word=pos_dict)
    for i in range(3):
        print(pos_lda.print_topic(i))

if __name__ == "__main__":
    # programmer_1()
    # programmer_2()
    # programmer_3()
    # programmer_4()
    # programmer_5()
    pass
