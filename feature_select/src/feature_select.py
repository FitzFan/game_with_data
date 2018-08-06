#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import sys
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

'''
    k_var : 方差选择需要满足的最小值
    pearson_value_k : 想剔除的feature个数
    vif_k ： 想剔除的feature个数
    wrapper_k ：想保留的feature个数
    penalty ： 'l1'正则或者'l2'正则
    C_0 : 惩罚力度
    
    sk_k: 想保留的feature个数
    score_fun: 特征选择的方法
        - chi2, 基于卡方进行特征选择
        - f_classif, 基于方差进行特征选择
        - mutual_info_classif, 基于互信息，用于分类问题
        - f_regression, 相关系数, 也可以用于分类问题
        - mutual_info_regression, 互信息度量 X 和 Y 共享的信息：它度量知道这两个变量其中一个，对另一个不确定度减少的程度, 用于回归问题

    - 玩法的不同
        - sk_select_fea()的返回可以直接作为model的输入，即为端对端；
        - 其它的func，返回的是column_name，需要先基于column_name筛选，再转为ndarray，最后喂给model；
        - 二者的不同之处在于使用的先后位置不一样。
    - **一个比较好的特征选择方法是：**
        - 每次构造的特征计算其在训练集和测试集上的均值和方差，保证分布相同，差不多则放入模型训练
'''


class feature_filter(object):
    def __init__(self, k_var=None, pearson_value_k=None, vif_k=None, wrapper_k=None, score_fun=None, sk_k=None, sel_model='lr', C_0=0.1, penalty='l2'):
        self.k_var = k_var
        self.pearson_value_k = pearson_value_k
        self.vif_k = vif_k
        self.wrapper_k = wrapper_k
        self.penalty = penalty
        self.C_0 = C_0
        self.sel_model = sel_model

        self.score_fun = score_fun
        self.sk_k = sk_k

    """基于sklearn的SelectKBest()进行特征选择, 作用是进行端对端的训练"""
    def sk_select_fea(self, data, label):
        base_score_func_list = ['chi2', 'f_classif', 'mutual_info_classif', 'f_regression', 'mutual_info_regression']
        if score_fun not in base_score_func_list:
            print 'score_fun is error, please select from following list:'
            print  base_score_func_list
        else:
            return SelectKBest(score_fun, k=self.sk_k).fit_transform(data, label)

    """下面这些func的作用是前期进行特征重要度分析，并非端对端。"""

    # 方差选择法
    """
    **** 待解决的问题：是否在使用方差法之前，先做一次归一化转换，对结果更有利？
        - 文杰知友的回答，maybe更能输出更好的结果，但还是要基于实验。因为拿到的数据永远都是真实数据的子集。
    """
    def var_filter(self, data):
        k = self.k_var
        var_data = data.var().sort_values()
        if k is not None:
            new_data = VarianceThreshold(threshold=k).fit_transform(data)
            return var_data, new_data
        else:
            return var_data

    # 线性相关系数衡量
    def pearson_value(self, data, label):
        k = self.pearson_value_k
        label = str(label)
        # k为想删除的feature个数
        Y = data[label]
        x = data[[x for x in data.columns if x != label]]
        res = []
        for i in range(x.shape[1]):
            data_res = np.c_[Y, x.iloc[:, i]].T
            cor_value = np.abs(np.corrcoef(data_res)[0, 1])
            res.append([label, x.columns[i], cor_value])
        res = sorted(np.array(res), key=lambda x: x[2], reverse=True)
        if k is not None:
            if k < len(res):
                new_c = []  # 保留的feature
                for i in range(len(res) - k):
                    new_c.append(res[i][1])
                return res, new_c
            else:
                print('feature个数越界～')
        else:
            return res

    # 共线性检验
    def vif_test(self, data, label):
        label = str(label)
        k = self.vif_k
        # k为想删除的feature个数
        x = data[[x for x in data.columns if x != label]]
        res = np.abs(np.corrcoef(x.T))
        vif_value = []
        for i in range(res.shape[0]):
            for j in range(res.shape[0]):
                if j > i:
                    vif_value.append([x.columns[i], x.columns[j], res[i, j]])
        vif_value = sorted(vif_value, key=lambda x: x[2])
        if k is not None:
            if k < len(vif_value):
                new_c = []  # 保留的feature
                for i in range(len(x)):
                    if vif_value[-i][1] not in new_c:
                        new_c.append(vif_value[-i][1])
                    else:
                        new_c.append(vif_value[-i][0])
                    if len(new_c) == k:
                        break
                out = [x for x in x.columns if x not in new_c]
                return vif_value, out
            else:
                print('feature个数越界～')
        else:
            return vif_value

    # Mutual Information
    def MI(self, X, Y):
        # len(X) should be equal to len(Y)
        # X,Y should be the class feature
        total = len(X)
        X_set = set(X)
        Y_set = set(Y)
        if len(X_set) > 10:
            print('%s非分类变量，请检查后再输入' % X_set)
            sys.exit()
        elif len(Y_set) > 10:
            print('%s非分类变量，请检查后再输入' % Y_set)
            sys.exit()
        # Mutual information
        MI = 0
        eps = 1.4e-45
        for i in X_set:
            for j in Y_set:
                indexi = np.where(X == i)
                indexj = np.where(Y == j)
                ijinter = np.intersect1d(indexi, indexj)
                px = 1.0 * len(indexi[0]) / total
                py = 1.0 * len(indexj[0]) / total
                pxy = 1.0 * len(ijinter) / total
                MI += pxy * np.log2(pxy / (px * py) + eps)
        return MI

    def mic_entroy(self, data, label):
        label = str(label)
        # k为想删除的feature个数
        x = data[[x for x in data.columns if x != label]]
        Y = data[label]
        mic_value = []
        for i in range(x.shape[1]):
            # 只对取值分布不超过10个的特征进行MI的特征选择
            if len(set(x.iloc[:, i])) <= 10: # x.iloc[:, i], 选取第i列的所有行
                res = self.MI(x.iloc[:, i], Y)
                mic_value.append([x.columns[i], res])
        mic_value = sorted(mic_value, key=lambda x: x[1])
        return mic_value

    # 递归特征消除法
    def wrapper_way(self, data, label):
        k = self.wrapper_k
        # k 为要保留的数据feature个数
        label = str(label)
        label_data = data[label]
        col = [x for x in data.columns if x != label]
        train_data = data[col]
        res = pd.DataFrame(
            RFE(estimator=LogisticRegression(), n_features_to_select=k, step=10).fit_transform(train_data, label_data))
        # estimator=LogisticRegression(), 使用LR进行训练，使用训练后的特征权重作为特征的重要度判断指标
        # n_features_to_select=k, 选择的特征f个数
        # step=10，每轮迭代删除10个特征

        res_c = []
        for i in range(res.shape[1]):
            for j in range(data.shape[1]):
                if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                    res_c.append(data.columns[j])
        res.columns = res_c

        return res

    # 基于model的方法
    def model_way(self, data, label):
        label_data = data[str(label)]
        col = [x for x in data.columns if x != str(label)]
        train_data = data[col]

        print 'Select Features From "%s" Model'%(self.sel_model)
        
        # 根据不同的选择，生成不同的model
        if self.sel_model == 'lr':
            model = LogisticRegression(penalty=self.penalty, C=self.C_0) # 参数C控制稀疏性，C越小所选择的特征越少
        elif self.sel_model == 'svm_lr':
            model = LinearSVC(penalty=self.penalty, C=self.C_0, dual=False) # 参数C控制稀疏性，C越小所选择的特征越少
        elif self.sel_model == 'gbdt':
            model = GradientBoostingClassifier()
        elif self.sel_model == 'extra':
            model = ExtraTreesClassifier()
        elif self.sel_model == 'lda':
            model = LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto')
        else:
            print 'select model is not available, please Select one of following models:'
            print 'lr, svm_lr, gbdt'
            sys.exit()

        # 根据所选择的model进行特征重要度排序
        res = pd.DataFrame(SelectFromModel(model).fit_transform(train_data, label_data))

        # 根据选择返回重要度比较高的feature
        res_c = []
        for i in range(res.shape[1]):
            for j in range(data.shape[1]):
                if (res.iloc[:, i] - data.iloc[:, j]).sum() == 0:
                    res_c.append(data.columns[j])
        res.columns = res_c

        return res

