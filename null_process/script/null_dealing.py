#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author  : Ryan Fan 
@E-Mail  : ryanfan0528@gmail.com
@Version : v1.0
"""

import pandas as pd
import numpy as np
import matplotlib as mat


class null_dealing(object):
    def __init__(self):
        print 'begin to do job of null_dealing'

	# 分位数填充缺失    
    def Key_Dealing(self, data_input, key_value=0.95):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = key_value
        for i in range(len(x.columns)):
            data1 = x.iloc[:, i].dropna(how='any')
            key = data1.quantile(y)
            data2 = x.iloc[:, i]
            data2 = data2.fillna(value=key)
            data2[data2 > key] = key
            data_union = pd.concat([data_union, data2], axis=1)
        return data_union

    # 以固定值填充缺失
    def Value_Dealing(self, data_input, Value):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = Value
        for i in range(len(x.columns)):
            key = y
            data2 = x.iloc[:, i]
            data2 = data2.fillna(value=key)
            data2[data2 > key] = key
            data_union = pd.concat([data_union, data2], axis=1)
        return data_union

    # 以众数填充缺失
    def Value_Mode(self, data_input, key_value=0.95):
        data_union = []
        data_union = pd.DataFrame(data_union)
        x = data_input
        y = key_value
        for i in range(len(x.columns)):
            data1 = x.iloc[:, i].dropna(how='any')
            data1 = data1.copy()
            key = data1.value_counts().argmax()
            data2 = data1.copy()
            key1 = data2.quantile(y)
            data3 = x.iloc[:, i]
            data3[data3 > key1] = key1
            data3 = data3.fillna(value=key)
            data_union = pd.concat([data_union, data3], axis=1)
        return data_union
