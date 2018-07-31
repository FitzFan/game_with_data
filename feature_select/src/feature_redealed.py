#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import os
import time
import sys
import re
import datetime
import pandas as pd 
import numpy as np


class data_filter(object):
    def __init__(self, data):
        self.data = data

    def nan_imputer(self, strategy='mean', null_base=0.2, fix_value_dict=None):
        """
        fix_value_dict: 
            - type：字典, key是column_name, value是fix_value
            - demo：{'ryan':0, 'fan':1}
        """

        # 获取所有的data_frame的列名
        columns_list = self.data.columns.values

        # 获取行数
        all_cnt = data.shape[0]

        # 遍历feature
        for column in columns_list:
            rate = np.isnan(np.array(self.data[column].values)).sum() / all_cnt
            # 对缺失比例不超过0.2的feature, 进行预设strategy的缺失填充
            if rate <= null_base:
                if fix_value == None:
                    self.data[column] = Imputer(missing_values='NaN', strategy=strategy, axis=0).fit_transform(self.data[[column]])
                else: # 对不同的缺失列，使用预设固定值填充
                    self.data.fillna(value=fix_value_dict, inplace=True)

            # 否则直接删除
            else:
                self.data.drop([column], axis=1, inplace=True)

        return self.data

    # 离群点盖帽
    def outlier_remove(data, limit_value=10, thre=1.5):
        # limit_value是最小处理样本个数set，当独立样本大于limit_value时，认为是连续性特征，存在异常值的可能性
        feature_cnt = data.shape[1]
        feature_change = []
        for i in range(feature_cnt):
            if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
                q1 = np.percentile(np.array(data.iloc[:, i]), 25)
                q3 = np.percentile(np.array(data.iloc[:, i]), 75)
                """
                复习一遍箱线图：
                - q3-q1为四分位差, 记为qr
                - 箱线图对异常值的定义是:
                    - 小于q1-1.5qr or 大于q3+1.5qr, 记为温和异常点（mild outliers）
                    - 小于q1-3.0qr or 大于q3+3.0qr, 记为极端异常点（extreme outliers）
                """
                top = q3 + thre * (q3 - q1)
                bottom = q1 - thre * (q3 - q1)
                data.iloc[:, i][data.iloc[:, i] > top] = top
                data.iloc[:, i][data.iloc[:, i] < bottom] = bottom

                feature_change.append(i)

            return data, feature_change




        