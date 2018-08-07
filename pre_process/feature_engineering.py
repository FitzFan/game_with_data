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
import warnings
from sklearn.preprocessing import StandardScaler

"""
feature engineering：
	- Feature types: raw features, statistic features, time-series features, cross features. 
	- statistic features needs to do Bayesian smooth
	- time-series features such as the number of installed app before clicktime, the number of installed app of the same type before clicktime
	- How to select cross features? use xgb features importance -> run xgb again to get updated features importance
	- How to code cross features? 1. Hash and onehotencoding  2. groupby -> transfer cross features to statistic features   
	- Gaussian_Normalization maybe can make output better
"""


def feature_engineer(object):
	def __init__(self, train, test, col_name):
		self.train = train
		self.test = test
		self.col_name = col_name

		# ignore all warnings
		warnings.filterwarnings('ignore')

	def Gaussian_Normalization(self):
		scaler = StandardScaler()
		for col in self.col_name:
			scaler.fit(list(self.train[col])+list(self.test[col]))
			self.train[col] = scaler.transform(self.train[col])
			self.test[col] = scaler.transform(self.test[col])

		return self.train, self.test







