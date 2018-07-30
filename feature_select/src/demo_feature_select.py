#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import pandas as pd
import sys
import numpy as np
from multi_feature_select import feature_filter


# load_data
path = '../data/feature_data.txt'
train_data = pd.read_table(path) # type is dataFrame

# 把null的数据去掉
index_col = [0]
index_col.extend(list(range(6, train_data.shape[1])))
train_data = train_data.iloc[:, index_col]


"""Begin To Select Features By following ways"""


# 方差,干掉方差小于100的列
model = feature_filter(k_var=100) # 返回的是tuple
# print model.var_filter(train_data)[0] # 打印各个feature的方差，默认是升序排列，第一行一般是是label


# 线性相关系数衡量，pearson_value_k：你想删除掉的feature个数
model = feature_filter(pearson_value_k=3)
res_list = model.pearson_value(train_data, 'label')[0] # 选择label作为因变量的列名, 返回的是list, 默认是根据coefficient升序排列
res_frame = pd.DataFrame()
res_frame['dependent variable'] = map(lambda x: x[0], res_list)
res_frame['features'] = map(lambda x: x[1], res_list)
res_frame['coefficient'] = map(lambda x: x[2], res_list)
# print res_frame


# 共线性检验，判断 feature 之间的相关性，剔除相关性较高的 feature
# vif_k想要剔除的feature个数, 个人建议是不传这个参数，按两两特征的相关系数进行倒排，看具体情况选择合适的vif_k
model = feature_filter(vif_k=3)
res_list = model.vif_test(train_data, 'label')[0]
res_frame = pd.DataFrame()
res_frame['feature_1'] = map(lambda x: x[0], res_list)
res_frame['feature_2'] = map(lambda x: x[1], res_list)
res_frame['coefficient'] = map(lambda x: x[2], res_list)
res_frame.sort_values(by='coefficient', ascending=False, inplace=True)
# print res_frame[0:100]


# Mutual Information检验
model = feature_filter()
res_list = model.mic_entroy(train_data, 'label')
res_frame = pd.DataFrame()
res_frame['feature'] = map(lambda x: x[0], res_list)
res_frame['Mutual Information'] = map(lambda x: x[1], res_list)
# print res_frame
# sys.exit()


# 递归特征消除法
# wrapper_k需要保留的特征个数
model = feature_filter(wrapper_k=3)
res_frame = model.wrapper_way(train_data, 'label')
# print res_frame.columns
# sys.exit()


# 基于模型的方法
"""
extre_tree是random_forest的变种，二者的区别如下：
- 随机森林应用的是 Bagging 模型，而 ET 是使用所有的训练样本得到每棵决策树，也就是每棵决策树应用的是相同的全部训练样本；
- 随机森林是在一个随机子集内得到最佳分叉属性，而 ET 是完全随机的得到分叉值，从而实现对决策树进行分叉的。
"""
sel_model = 'lda' # 只接受以下取值：lr, svm_lr, gbdt, extra_tree, lda(线性判别)
penalty = 'l2'
C_0 = 0.1
model = feature_filter(sel_model=sel_model, penalty=penalty, C_0=C_0)
res_frame = model.model_way(train_data, 'label')
print res_frame.columns
sys.exit()

