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