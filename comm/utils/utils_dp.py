#!/usr/bin/env python
# -*- coding: utf-8 -*-
# This is a module that contains methods for data processing inherented from my work in DEXTRA
# No update since Nov 2015
# Removing outliners:
#for col in ['C18','C20','C21']:
#    # keep only the ones that are within +3 to -3 standard deviations in the column col,
#    train = train[np.abs(train[col]-train[col].mean())<=(3*train[col].std())]
#    # or if you prefer the other way around
#    train = train[~(np.abs(train[col]-train[col].mean())>(3*train[col].std()))]
import csv as csv
import math
import time
from datetime import datetime as dt

import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit as sss




class R_Python_Interface(object):
	"""for Python to call R functions/packages. It is first created to clean Unilever survey data for post Unilever data challenges,
	but some methods can be actually used for general purpose, such as install missing r packages.
	"""

	def __init__(self, packnames):
		"""Initialise the object through checking the required packages and install them if needed.

		Input ->
		packnames :    tuple of required r packages
		"""
		# import rpy2's package module
		import rpy2.robjects.packages as rpackages
		if all(rpackages.isinstalled(x) for x in packnames):
			have_tutorial_packages = True
		else:
			have_tutorial_packages = False
		if not have_tutorial_packages:
			# import R's utility package
			utils = rpackages.importr('utils')
			# select a mirror for R packages
			utils.chooseCRANmirror(ind=1) # select the first mirror in the list
		if not have_tutorial_packages:
			# R vector of strings
			from rpy2.robjects.vectors import StrVector
			# file
			packnames_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
			if len(packnames_to_install) > 0:
				utils.install_packages(StrVector(packnames_to_install))


	def call_r_func(self):
		"""Sample method to demonstrate how to write a method to call r function/packages."""
		pass
		

class R_Python_Unilever(R_Python_Interface):
	"""Employ R functions/packages in python script to clean the data of Unilever challenge. It inherits from the general class 
	<R_Python_Interface>, and adds more sepcific methods for Unilever data cleaning.
	"""

	def __init__(self, packnames, path):
		"""Initialise the object through checking available packages

		Input ->
		packnames :    tuple of required r packages
		path :         string of full path to the spss file, e.g. "/Volumes/Data/Google Drive/DEX:DEXTRA/02 Global Lead Users/5. Unilever/6 Data/1 - 110778/110778.sav"
		"""
		super(R_Python_Unilever, self).__init__(packnames) # Call parent class to initialise the required packages
		self._file_path = path

	def read_spss_to_df(self):
		"""Use R functions to read SPSS files

		Input ->
		NULL
		====================================================================================================
		Output ->
		Return a tuple of a python DataFrame and an np array of descriptions of column names (i.e. features descriptions)
		"""
		from rpy2.robjects import r
		from string import Template
		from rpy2.robjects import pandas2ri
		import unicodedata
		file_location = self._file_path # or "./1 - 110778/110778.sav"
		file_location_csv = file_location[:-4] + ".csv"
		r_code = Template('''
		library(foreign)
		library(plyr)

		df <- read.spss ("$origin_file", to.data.frame=TRUE)
		desc <- attr(df,"variable.labels")
		write.csv(df, file="$output_file", na="")
		''')
		r_code = r_code.substitute(origin_file=file_location, output_file=file_location_csv) # Substitute input and output file with variables presented in python
		r(r_code) # Run the above r code in r global environment

		df = pandas2ri.ri2py(r('df')) # convert from r data frame into pandas data frame
		df = df.applymap(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore') if type(x) == unicode else x) # Translate unicode encoding into ascii encoding

		desc = pandas2ri.ri2py(r('desc')) # convert into python variable
		for j, ele in enumerate(desc):
			if type(desc[j]) == np.unicode_:
				desc[j] = str(unicodedata.normalize('NFKD', desc[j]).encode('ascii','ignore')) # http://stackoverflow.com/questions/1207457/convert-a-unicode-string-to-a-string-in-python-containing-extra-symbols
		desc = desc.astype(np.string_)
		return df, desc


def read_r_to_python(path_I, path_II):
	"""Read variables stored in R data format, and then convert it into Python data frame or array
	This method is DEPRECIATED due to class <R_Python_Unilever>"""
	from rpy2.robjects import r
	from rpy2.robjects import pandas2ri
	import unicodedata

	tmp = r.readRDS(path_I) # read from r file
	df = pandas2ri.ri2py(tmp) # convert into pandas data frame
	df = df.applymap(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore') if type(x) == unicode else x) # Translate unicode encoding into ascii encoding

	tmp = r.readRDS(path_II)
	desc = pandas2ri.ri2py(tmp) # convert into python variable
	for j, ele in enumerate(desc):
		if type(desc[j]) == np.unicode_:
			desc[j] = str(unicodedata.normalize('NFKD', desc[j]).encode('ascii','ignore')) # http://stackoverflow.com/questions/1207457/convert-a-unicode-string-to-a-string-in-python-containing-extra-symbols
	desc = desc.astype(np.string_)
	return df, desc

class UnileverDataCleaning(object):
	"""Contains the pipeline of data cleaning for Unilever raw data.
	Run the methods in class R_Python_Unilever first to get df and desc,
	then feed the instance of this class with df and desc
	"""
	def __init__(self, path, df, desc):
		"""Initialise the instance for given dataset.

		Input ->
		path :  string of full path to the spss file, e.g. "/Volumes/Data/Google Drive/DEX:DEXTRA/02 Global Lead Users/5. Unilever/6 Data/1 - 110778/110778.sav"
		df :    pandas DataFrame, obtained from method <read_spss_to_df> in class <R_Python_Unilever>
		desc :  1-D np array, obtained from method <read_spss_to_df> in class <R_Python_Unilever>
		=========================================================================================================
		Output ->
		Initialise some variables for late usage
		"""
		self._path = path
		self._df = df
		self._desc = desc
		self._dict = None
		self._dict_new = None
		self._process_ctl = [0]*6 # create an array of 0s to monitor the current process stage. Totally 5 methods involved in the pipeline process.

	def get_df(self):
		"""Return the input dataframe"""
		return self._df

	def get_desc(self):
		"""Return the input 1-D np array"""
		return self._desc

	def get_dict(self):
		"""Match the descriptions of column name and its description, and return it as a dictionary
		Then dictionary is transformed into dataframe, with index being the key

		Output ->
		self._dict :  a dataframe transformed from dictionary, whose key is column name of data, and value is the corresponding descriptions
		"""
		if not self._process_ctl[0]: # The 1st step of the pipeline, so first element in _process_ctl is used to monitor it.
			cln_name = self._df.columns.values
			cln_desc = self._desc
			tmp_dict = dict(zip(cln_name, cln_desc))
			self._dict = pd.Series(tmp_dict, name='Desc1')

			self._process_ctl[0] = 1
		return self._dict

	def remove_colon(self):
		"""Remove some columns based on their descriptions. Basically, some descriptions contain colon, which means they are
		re-directed from other questions, and those columns shall be removed

		Output ->
		self._dict_new :   a new dataframe, deep copied from self._dict; columns with colon are removed as compared to original dateframe
		"""
		if not self._process_ctl[1]: # The 2nd step of the pipeline
			self.get_dict() # Run the previous step to make sure the pre-requirement is fulfilled

			self._dict_new = self._dict.copy()
			index = self._dict_new.index.values
			for idx in index:
				if ":" in self._dict_new[idx]:
					self._dict_new = self._dict_new.drop([idx])

			self._process_ctl[1] = 1


	@staticmethod
	def refine_pattern(str_tmp):
		"""Extract information from the descriptions, remove redundant words

		Input ->
		str_tmp :  a string to be cleaned
		=====================================================================================================================
		Return cleaned string if some partten is matached, otherwise return the original string
		"""
		import re
		match = re.search(r'(^\w+\d+\w*\s)(.*)', str_tmp) # ^ -> start of string, \w -> words, \d -> digits, \s -> whitespace
		if match:
			return match.group(2)
		else:
			return str_tmp

	@staticmethod
	def refine_pattern2(str_tmp):
		"""Extract information from the descriptions, remove redundant words

		Input ->
		str_tmp :  a string to be cleaned
		=====================================================================================================================
		Return cleaned string if some partten is matached, otherwise return the original string
		"""
		import re
		match = re.search(r'(.*)(\(\w+\d+\w*_\d+\))(\s*.*)', str_tmp) # \w -> words, \d -> digits, \s -> whitespace
		if match:
			return match.group(1) + match.group(3)
		else:
			return str_tmp

	def refine_desc(self):
		"""self.remove_colon -> self.refine_desc: work on the descriptions so as to remove redundant symbols and standardize the descriptions"""
		if not self._process_ctl[2]: # The 3rd step of the pipeline
			self.remove_colon()

			self._dict_new = pd.concat([self._dict_new, self._dict_new], axis=1, keys=['Desc1','Desc2'])
			index = self._dict_new.index.values
			for idx in index:
				self._dict_new['Desc2'][idx] = self.refine_pattern(self._dict_new['Desc2'][idx])

			self._process_ctl[2] = 1
#		return self._dict_new
			'''
			for key in self._dict_new.keys():
				self._dict_new[key] = self.refine_pattern(self._dict_new[key])
			'''

	def refine_desc2(self):
		"""self.remove_colon -> self.refine_desc -> self.refine_desc2: work on the descriptions so as to remove redundant symbols and standardize the descriptions"""
		if not self._process_ctl[3]: # The 4th step of the pipeline
			self.refine_desc()

			self._dict_new = pd.concat([self._dict_new, self._dict_new['Desc2']], axis=1)
			self._dict_new.columns = ['Desc1','Desc2', 'Desc3']
			index = self._dict_new.index.values
			for idx in index:
				self._dict_new['Desc3'][idx] = self.refine_pattern2(self._dict_new['Desc3'][idx])

			self._process_ctl[3] = 1
#		return self._dict_new

	def drop_duplicate(self):
		"""self.remove_colon -> self.refine_desc -> self.refine_desc2 -> drop_duplicate: after removing noises from descriptions, some descriptions 
		become duplicated. Even their content is different, but we cannot distinguish them through descriptions, so we discard them"""
		if not self._process_ctl[4]: # The 5th step of the pipeline
			self.refine_desc2()

			self._dict_new.drop_duplicates(subset='Desc3', keep=False, inplace=True)

			self._process_ctl[4] = 1

	def remove_punctuation(self):
		"""self.remove_colon -> self.refine_desc -> self.refine_desc2 -> drop_duplicate -> remove_punctuation: replace punctuation like ._? 
		with whitespace, and change all words into lower cases"""
		import string
		if not self._process_ctl[5]: # The 6th step of the pipeline
			self.drop_duplicate()

			map_table = string.maketrans(string.punctuation, len(string.punctuation)*" ") # Create the table for replacing punctuation with whitespace
			self._dict_new = pd.concat([self._dict_new, self._dict_new['Desc3']], axis=1)
			self._dict_new.columns = ['Desc1','Desc2', 'Desc3', 'Desc4']
	#		self._dict_new.info()
			index = self._dict_new.index.values
			for idx in index:
				self._dict_new['Desc4'][idx] = self._dict_new['Desc4'][idx].translate(map_table).lower()
				self._dict_new['Desc4'][idx] = " ".join(self._dict_new['Desc4'][idx].split())

			self._process_ctl[5] = 1


	def save_dict(self, path_dict = None):
		"""Save column_name - column_description dictionary into csv file"""
		if self._dict is None:
			self.get_dict()
		if path_dict is None:
			path_dict = self._path[:-4] + "_dict.csv"
		self._dict.to_csv(path_dict)#, encoding='utf-8')

	def save_dict_new(self, path_dict = None):
		"""Save the new dictionary into csv file"""
		if self._dict_new is None:
			self.remove_colon()
		if path_dict is None:
			path_dict = self._path[:-4] + "_dict_new.csv"
		self._dict_new.to_csv(path_dict)


class create_files_for_participant(object):
	"""Contains various methods used to create sample submissions of challenges for participants.

	Different types of challenges shall deserve different methods for creating sample submission.
	"""
	def sample_submission(self, header, data, cln_name):
		"""Create sample submission file for challenge pariticipants from the test dataset

		Input -> 
		header :   np array or list of header, i.e. column names
		data :     np array of the test dataset
		cln_name : column name of the id column
		=============================================================
		Output ->
		2-D np array of dimension n X 2, where n is the number of rows, 2 columns with one column id, the other randomly assigned results.
		=============================================================
		Note ->
		This is to create sample submission for binary classification problems evaluated using AUC or Log Loss.
		This method is actually DEPRECIATED because it doesn't make use of the package pandas.
		"""
		num_rows = np.size(data[:,0]) # number of rows
		cln_num = np.where(header == cln_name)[1][0]
		id_column = data[:,cln_num].reshape((num_rows,1)) # extract the id column of the dataset
		random_assignment = np.random.uniform(0.0, 1.0, size = (num_rows,1)).astype(str) # -> This is to generate uniform float value between 0 and 1
	#!	print random_assignment
	#!	print file_paths
		return np.hstack((id_column, random_assignment))

class data_clean(object):
	'''
	This class include methods for data cleaning. The main packages used here are pandas and numpy.
	'''
	def read_remove_clns(self, path, cln_list):
		"""This method is not actually useful. So it is DEPRECIATED."""
		df = pd.read_csv(path, header = 0) # Read the csv file into data frame
		return df.drop(cln_list, axis = 1) # Remove columns in the list cln_list

	def read_remove_rows(self, path_I, path_II, cln):
		'''This function reads 2 csv files into data frames df_1 and df_2, and for the common column 'cln', remove rows
		in df_1 with entry appearing in df_2

		:return -> new data frame of df_1 with rows that are in df_2 removed
		'''
		with open(path_I) as f1, open(path_II) as f2:
			df1 = pd.read_csv(path_I, header = 0)
			df2 = pd.read_csv(path_II, header = 0)

			arr_1 = df1[cln].values.astype(str)
			arr_2 = df2[cln].values.astype(str)

			arr_1 = np.char.lower(arr_1)
			arr_2 = np.char.lower(arr_2)

			idx_true_false = np.invert(np.in1d(arr_1, arr_2))
			df_new = df1[idx_true_false]
		return df_new

class datetimeparse(object):
	'''This class contains the method to parse time string in dataframe, and separate time into year month day weekday time and etc.
	'''
	def __init__():
		pass

	@classmethod
	def split_beta(cls, df, time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day'], special_columns = None):
		'''This method is the prototype, very slow. Please check the new version <split2> below
		'''
		num_clns = len(columns)
		num_rows = df.size
		cln_name = df.name
		new_cln_names = map(lambda x: cln_name + '_' + x, columns)
		new_df = pd.DataFrame(np.zeros([num_rows, num_clns]), columns = new_cln_names)
		dict_time_retrive = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5, 'weekday':6, 'day_in_year':7}
		# datetime obj turple contains: tm_year=2014, tm_mon=8, tm_mday=31, tm_hour=22, tm_min=15, tm_sec=0, tm_wday=6, tm_yday=243, tm_isdst=-1
#!		print new_cln_names
#!		print new_df.info()
		# The for loop takes 3-5 mins, too long, need improvement, such as using apply in data frame
		if special_columns:
			num_clns_2 = 1
			new_cln_names_2 = ['total_seconds']
			new_df_2 = pd.DataFrame(np.zeros([num_rows, num_clns_2]), columns = new_cln_names_2)
			for i, date_time in enumerate(df):
				date_time_obj = dt.strptime(df[i], time_str_format)
				date_time_tuple = date_time_obj.timetuple()
				for cln, cln2 in zip(new_cln_names, columns):
					new_df[cln][i] = date_time_tuple[dict_time_retrive[cln2]]

				new_df_2[i] = (date_time_obj - special_columns).total_seconds()
			new_df = pd.concat([new_df, new_df_2], axis = 1)
		else:
			for i, date_time in enumerate(df):
				date_time_obj = dt.strptime(df[i], time_str_format)
				date_time_tuple = date_time_obj.timetuple()
				for cln, cln2 in zip(new_cln_names, columns):
					new_df[cln][i] = date_time_tuple[dict_time_retrive[cln2]]

		return new_df

	@classmethod
	def split(cls, df, time_str_format = "%m/%d/%Y %H:%M", columns = ['month', 'day'], special_columns = None):
		'''This classmethod is an upgraded one of <split>, the  main change is that the for for loop is replaced with DataFrame.apply
		After test, it is proved to be much faster than the prototype
		'''
		num_clns = len(columns)
		num_rows = df.size
		cln_name = df.name
		new_cln_names = map(lambda x: cln_name + '_' + x, columns)
		new_df = pd.DataFrame(np.zeros([num_rows, num_clns]), columns = new_cln_names)
		dict_time_retrive = {'year':0, 'month':1, 'day':2, 'hour':3, 'minute':4, 'second':5, 'weekday':6, 'day_in_year':7}
		# datetime obj turple contains: tm_year=2014, tm_mon=8, tm_mday=31, tm_hour=22, tm_min=15, tm_sec=0, tm_wday=6, tm_yday=243, tm_isdst=-1
#!		print new_cln_names
#!		print new_df.info()
		# The for loop takes 3-5 mins, too long, need improvement, such as using apply in data frame
		df = df.apply(lambda x: dt.strptime(x, time_str_format))
		df_tuple = df.apply(lambda x: x.timetuple())
		if special_columns:
			num_clns_2 = 1
			for cln, cln2 in zip(new_cln_names, columns):
				new_df[cln] = df_tuple.apply(lambda x: x[dict_time_retrive[cln2]])

			new_df_2 = df.apply(lambda x: (x - special_columns).total_seconds())
			new_df_2.name = cln_name + '_' + 'total_seconds'
			new_df = pd.concat([new_df, new_df_2], axis = 1)
		else:
			for cln, cln2 in zip(new_cln_names, columns):
				new_df[cln] = df_tuple.apply(lambda x: x[dict_time_retrive[cln2]])

		return new_df

class evaluation_metric(object):
	"""Collections of evaluation metric implemented using python."""
	def LogarithmicLoss(self, submission, public, private):
		'''
		This method is for evaluating participants' submission using LogLoss.


		Input ->
		submission : np array of submission
		public :     np array of the backend array for public score
		private :    np array of the backend array for private score
		========================================================================================
		Output ->
		return a tuple of public and private score
		'''
		dict_submission = dict(zip(submission[:, 0], submission[:, 1]))
		dict_public = dict(zip(public[:, 0], public[:, 1]))
		dict_private = dict(zip(private[:, 0], private[:, 1]))

		sum_public = 0.0
		for key in dict_public.keys():
			truth = int(dict_public[key])
			if (truth == 1):
				prediction = max(float(dict_submission[key]), pow(10, -15))
			else:
				prediction = max(1.0 - float(dict_submission[key]), pow(10, -15))
			sum_public = sum_public + math.log10(prediction)

		sum_private = 0.0
		for key in dict_private.keys():
			truth = int(dict_private[key])
			if (truth == 1):
				prediction = max(float(dict_submission[key]), pow(10, -15))
			else:
				prediction = max(1.0 - float(dict_submission[key]), pow(10, -15))
			sum_private = sum_private + math.log10(prediction)

		return - sum_public/len(dict_public.keys()), - sum_private/len(dict_private.keys())

	def LogLoss(self, truth, prediction):
		'''
		Evaluate the prediction against truth using Log Loss metric

		Input ->
		truth :      np array of true value, 2 columns, row id and corresponding prediction
		prediction : np array of predicted value, 2 columns, row id and corresponding prediction
		===========================================================================
		Output ->
		Return LogLoss evaluation score
		'''
		dict_truth = dict(zip(truth[:, 0], truth[:, 1]))
		dict_predicted = dict(zip(prediction[:, 0], prediction[:, 1]))

		score = 0.0
		for key in dict_truth.keys():
			truth = int(dict_truth[key])
			if (truth == 1):
				predicted = min(1, max(float(dict_predicted[key]), pow(10, -15)))
			else:
				predicted = min(1, max(1.0 - float(dict_predicted[key]), pow(10, -15)))
			score = score + math.log10(predicted)
		return - score/len(dict_truth.keys())

	def LogLoss_II(self, truth, prediction):
		'''
		Evaluate the prediction against truth using Log Loss metric

		Input ->
		truth :      np array of true value, 1 column
		prediction : np array of predicted value, 1 column
		===========================================================================
		Output ->
		Return LogLoss evaluation score
		'''
		N = len(truth)

		score = 0.0
		for i in xrange(N):
			truth_value = int(truth[i])
			if (truth_value == 1):
				predicted = min(1, max(float(prediction[i]), pow(10, -15)))
			else:
				predicted = min(1, max(1.0 - float(prediction[i]), pow(10, -15)))
			score = score + math.log10(predicted)
		return - score/N

	@classmethod
	def Precision(cls, truth, prediction):
		'''This metric is used for calculate precision of binary classification, mainly for implementing on DEXTRA
		'''
		pass

	@classmethod
	def Accuracy(cls, truth, prediction):
		pass

	@classmethod
	def F1(cls, truth, prediction):
		pass

	@classmethod
	def AUC(cls, truth, prediction):
		pass

class engineer_feature(object):
	'''Contains various methods used for feature engineering.
	'''
	def __init__(self, df_train, df_test = None):#, df, param = {'pd_describe': True, 'zero': False}):
		'''Initialise parameters for feature engineering, mainly determine which feature to be engineered
		'''
		frames = [df_train, df_test]
		self.df = pd.concat(frames, keys=['x', 'y'])
		self.method_list = []
		self.df_final = None
#		_key_lst = ['pd_describe', 'zero']
#		for key in key

	def pd_feature(self):
		'''This method will add features to the data frame using the statistics of pandas built-in method
		lessons learned from pandas apply method: I. apply usually assume the data type is object, which is
		often not desirable. DO CONVERT it to what you want, otherwise you get wrong result. 
		II. Series is quite differ from dataframe, at least in terms of shape. <apply> assumes
		the passed/returned data is series of shape (n,), NOT dataframe of shape (n,1)!!!
		'''
		self.df_pd = self.df.apply(lambda x: x.astype('float64').describe(), axis=1)
		self.method_list.append(self.df_pd)
		return self.df_pd.ix['x'], self.df_pd.ix['y']

	def _get_corr(self, df_series, y):
		'''Calculate the covariance between feature/column and label

		Parameters
		----------
		df_series : pandas.series, column of training data
		y : 1-D numpy array, label of training data
		'''
		x = df_series.values
		idx = np.isnan(x)
		idx = np.invert(idx)
#		print idx
		x = x[idx]
#		print x
		y = y[idx]

		n = len(x) + 0.0
		mean1 = sum(x) / n
		mean2 = sum(y) / n

		variance1 = 0
		variance2 = 0

		covariance = 0
		for i in range(int(n)):
			a = x[i] - mean1
			b = y[i] - mean2
			covariance += a*b
			variance1 += a*a
			variance2 += b*b
		correlation = covariance/math.sqrt(variance1*variance2)

		return correlation

	def corr_feature(self, y):
		'''Evaluate the covariance of each feature/column with respect to label

		Parameters
		----------
		y : 1-D numpy array, label of training data
		'''
		df_train = self.df.ix['x']
		df_corr = df_train.apply(self._get_corr, axis=0, args=(y,))
		print df_corr

	def aggregate_features(self):
		'''Concatenates all the features created here, and return the 
		'''
		self.df_final = self.df
		for df in self.method_list:
			self.df_final = pd.concat([self.df_final, df], axis = 1)
		return self.df_final.ix['x'], self.df_final.ix['y']

	def to_csv(self, path_train='./train_data_with_new_features.csv', path_test='./test_data_with_new_features.csv', train_index=None, train_label=None, test_index=None, index=False):
		"""Save data with new features added into csv files for late process."""
		if self.df_final is not None:
			df_train = pd.concat([train_index, self.df_final.ix['x'], train_label], axis=1)
			df_test = pd.concat([test_index, self.df_final.ix['y']], axis=1)
			df_train.to_csv(path_train, index = index)
			df_test.to_csv(path_test, index = index)
#			self.df_final.to_csv(path, index = index)
		else:
			self.aggregate_features()
			self.to_csv(path_train, path_test, train_index, train_label, test_index, index)
#			raise ValueError('Call method aggregate_features before call this method')

class local_evaluation(object):
	"""Collections of evaluation metric used for local evaluation.

	For example, to evaluate the performance of a binary classification, 4 evaluation metrics are combined
	and displayed in a row here to give a straightforward view of its overall performance.
	"""
	@classmethod
	def bi_classification_4(cls, y_true, y_pred, y_pred_prob = None):
		"""Evaluate the performance of binary classification using 4 metrics together.

		Input ->
		y_true :       1-D np array of true values of either 0 or 1
		y_pred :       1-D np array of predicted values of either 0 or 1
		y_pred_prob :  1-D np array of predicted values in the form of probability, ranging from 0 to 1. It's mainly used for metric AUC
		================================================================================
		Output ->
		Display all the 4 evaluated values in a row
		"""
		from sklearn import metrics
		recall = metrics.recall_score(y_true, y_pred)
		precision = metrics.precision_score(y_true, y_pred)
		f1 = metrics.f1_score(y_true, y_pred)
		if y_pred_prob is not None:
			auc = metrics.roc_auc_score(y_true, y_pred_prob)
		else:
			auc = metrics.roc_auc_score(y_true, y_pred)
		print "Recall is %f, Precision is %f, f1 score is %f, and AUC is %f" % (recall, precision, f1, auc)

class lookup(read_write):
	"""Collection of methods used for mask data and validation of mask.

	Basically, the idea is to first create a dictionary for original values and masked values, and
	then mask original values according to the dictionary. Additional methods are created to validate
	whether the masking is done correctly.
	"""
	def dict_create(self, keys, prefix = None, file_path = None, shuff = False):
		"""Create a dictionary to map original values to new values, for two purposes: one is to mask, the other is to transform long, unreadable values into human readable values
		One example of mapping is: "kajfksajgjhkjaghg" -> "MV007"

		Input ->
		keys :       1-D np array of strings, old values
		prefix :     string used as prefix of new values, e.g. "MV" in the example above
		file_path :  a string of file path, used to store the dictionary into a csv file
		shuff :      True of Flase; if true, the keys passed in will be shuffed and randomized. Default is False
		====================================================================================
		Output -> return the created dictionary, and write the dictionary into a csv file.
		"""
		if shuff:
			np.random.shuffle(keys) # random shuffle the array
		if prefix:
			values = map(lambda x: prefix.format(x), xrange(1,len(keys)+1))
		else:
			values = xrange(1,len(keys)+1)
		list_of_turples = zip(keys, values)
		dictionary = dict(list_of_turples)
		if file_path:
			read_write().array_to_csv(list_of_turples, file_path, np.array(['orginal_id', 'new_id']))
		return dictionary

	def dict_validate(self, origin, descent, dictionary, cln_name):
		'''
		This method take 3 files as input: origin, descent and dictionary, and then check whether the lookup is totally correct;
		Here we assume that the order of row does not change after lookup

		Input ->
		origin :        a string of the location of original data file
		descent :       a string of the location of new data file after mapping
		dictionary :    a string of the location of the dictionary file
		cln_name :      a string, column name of the mapped column
		================================================================================
		Output ->
		Print customized error message for mis-matched origin - new value pairs
		'''
		(header, arr_origin) = self.tsv_to_array(origin)
		(header, arr_descent) = self.csv_to_array(descent, "rU")
		cln_id = np.where(header == cln_name)[1][0]
		print cln_id
		(header, arr_dict) = self.csv_to_array(dictionary)
		dict_dict = dict(zip(arr_dict[:, 0], arr_dict[:, 1]))
		for row_origin, row_descent in zip(arr_origin, arr_descent):
			if dict_dict[row_origin[cln_id]] != row_descent[cln_id]:
				print "Error"
				print "Original id is %s, descent id is %s: " % (row_origin[cln_id], row_descent[cln_id])
				print "The dictionary of original id %s is %s" % (row_origin[cln_id], dict_dict[row_origin[cln_id]])
				print

	def dict_validate_II(self, origin, descent, dictionary, country_ctl = False):
		'''
		This method take 2 files as input: origin and descent; and a dictionary of all dictionaries of masked IDs, 
		and then check whether the lookup is totally correct;
		Here we assume that the order of row does not change after lookup

		Input ->
		origin :        a string of the location of original data file
		descent :       a string of the location of new data file after mapping
		dictionary :    a dictionary of all dictionaries of masked IDs
		country_ctl :   True or False, used to control whether to check the masking of column named "country". This is customized to Rakuten data challenge dataset, because some columns with name "country" are masked, and some are not.
		================================================================================
		Output ->
		Print customized error message for mis-matched origin - new value pairs
		================================================================================
		Note ->
		This method differs from the above this way: the above requires the original file, masked file and a specific column, and then do the check;
		the current requires the original file, masked file and a dictionary of all dictionaries of masked IDs, so it can check all the columns in the
		two given files, and no need human to specify column by column as the above. So it is much more efficient.
		'''
		(header_1, arr_origin) = self.tsv_to_array(origin)
		(header_2, arr_descent) = self.csv_to_array(descent)
		for cln_name in header_1.flatten():
			if country_ctl:
				true_false = (cln_name in dictionary.keys()) & (cln_name != "country")
			else:
				true_false = cln_name in dictionary.keys()
			if true_false:
				dict_dict = dictionary[cln_name] # This is to extract the dictionary for this particular column
				cln_id_1 = np.where(header_1 == cln_name)[1][0]
				cln_id_2 = np.where(header_2 == cln_name)[1][0]
				print cln_name
				print cln_id_1
				print cln_id_2
				for row_origin, row_descent in zip(arr_origin, arr_descent):
					if dict_dict[row_origin[cln_id_1]] != row_descent[cln_id_2]:
						print "Error"
						print "Original id is %s, descent id is %s: " % (row_origin[cln_id_1], row_descent[cln_id_2])
						print "The dictionary of original id %s is %s" % (row_origin[cln_id_1], dict_dict[row_origin[cln_id_1]])
						print


class slicing(object):
	"""Collection of methods for generating chunks to slice an array.
	The idea and syntax are gotten from stackoverflow.
	"""
	def chunks(self, array, granular, arr_len):
		"""Create a generator to yield a fixed length array from a big array

		Input ->
		array :      the big parent array
		granular :   the size of the sliced sub-array
		arr_len :    length of the array
		=================================================================================
		Output ->
		a generator that generates sub-array
		"""
		for i in xrange(0, arr_len, granular):
			yield array[i:i+granular]

	def chunks_arr(self, array, granular_array):
		"""Create a generator to yield sub-arrays from a big array. The size of the sub-arrays are determined by an array containing the length info of sub-arrays

		Input ->
		array :            the big parent array
		granular_array :   the array containing the length info of all sub-arrays
		================================================================================================
		Output ->
		a generator that generates sub-arrays accordingly
		"""
		index = 0
		for i in xrange(0, len(granular_array)):
			yield array[index : index + granular_array[i]]
			index = index + granular_array[i]

class sort_arr(object):
	"""Collection of customized sorting methods."""
	def one_index_sort(self, array, cln):
		"""Sort the whole 2-D array according to a given column in the array.

		Input ->
		array : 2-D array
		cln :   column id of the given column to be sorted by
		=======================================================================================
		Output ->
		return the sorted array
		"""
		return array[array[:, cln].astype(int).argsort(),:]

	def two_index_sort(self, array, cln_1, cln_2, reverse_1 = False, reverse_2 = False):
		"""Sort the whole 2-D array according to two given columns in the array.

		Input ->
		array :      2-D array
		cln_1 :      1st column id of two given columns to be sorted by
		cln_2 :      2nd column id of two given columns to be sorted by
		reverse_1 :  True or False for 1st column, True for descending, and False for ascending; Default False
		reverse_2 :  True or False for 2nd column, True for descending, and False for ascending; Default False
		=======================================================================================
		Output ->
		return the sorted array
		"""
		order_1 = 1
		order_2 = 1
		if reverse_1:
			order_1 = -1
		if reverse_2:
			order_2 = -1
		ind = np.lexsort((array[:, cln_2].astype(int) * order_2, array[:, cln_1].astype(int) * order_1))
		return array[ind,:]

class stratification(object):
	"""Collection of methods used for stratification."""
	@classmethod
	def one_cln1(cls, array, train = 0.8):
		"""Split a 1-D array into 2 segments, one for training, the other for testing.

		Input ->
		array :   1-D array, the column to be stratified, taken from the whole 2-D array
		train :   the ratio of training data, ranging from 0 to 1
		============================================================================================
		Output ->
		Return a tuple of 2 arrays, one contains the index of training data, and the other contains index of testing data
		"""
		random_state = 17#int(time.time()%60)
		length = len(array)
		arr = np.arange(length)
		np.random.shuffle(arr)
		
		trunc_inx = round(length*train)
		train_index = arr[:trunc_inx]
		test_index = arr[trunc_inx:]
		return (train_index, test_index)

	@classmethod
	def one_cln2(cls, array, train = 0.8, random_state = 9):
		"""Split a 1-D array into 2 segments, one for training, the other for testing.

		Input ->
		array :   1-D array, the column to be stratified, taken from the whole 2-D array
		train :   the ratio of training data, ranging from 0 to 1
		============================================================================================
		Output ->
		Return a tuple of 2 arrays, one contains the index of training data, and the other contains index of testing data
		"""
		#int(time.time()%60)
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train), random_state = random_state)
		print 'random_state is ', random_state
		for train_index, test_index in s3:
			pass
		return (train_index, test_index)

	@classmethod
	def one_cln3(cls, array, train = 0.6, cross = 0.2):
		"""Split a 1-D array into 3 segments, one for training, one for cross validation and the third for testing.

		Input ->
		array :   1-D array, the column to be stratified, taken from the whole 2-D array
		train :   the ratio of training data, ranging from 0 to 1
		cross :   the ratio of cross validation data, ranging from (1 - train) to 1
		============================================================================================
		Output ->
		Return a tuple of 3 arrays, one contains the index of training data, one contains the index of cross validation data, and the third contains index of testing data
		"""
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train - cross), train_size = train, random_state = int(time.time()%60))
		for train_index, test_index in s3:
			pass
		index = np.array(xrange(array.size))
		cross_index = np.setdiff1d(index, np.concatenate((train_index, test_index)), assume_unique = True)
		np.random.shuffle(cross_index)
		return (train_index, cross_index, test_index)

	# The method below select train/test/validation datasets based on stratification on one column
	# It returns a turple of 3 arrays containing the index for train/test/validation datasets respectively
	def one_cln(self, array, train = 0.6, cross = 0.2):
		"""Same as method <one_cln3>, the difference is that one_cln3 is a class method."""
		s3 = sss(array, n_iter = 1, test_size = (1.0 - train - cross), train_size = train, random_state = int(time.time()%60))
		for train_index, test_index in s3:
			pass
		index = np.array(xrange(array.size))
		cross_index = np.setdiff1d(index, np.concatenate((train_index, test_index)), assume_unique = True)
		np.random.shuffle(cross_index)
		return (train_index, cross_index, test_index)

	def two_cln_fake(self, header, array, clns, train = 0.6, cross = 0.2):
		pass

	def data_randomnize(self, data_extracted, split_ratio):
		"""Randomnize the given dataset, and then split them according the split ratio

		Input ->
		data_extracted :  2-D array of the whole dataset
		split_ratio :     1-D array of size 2, 1st element is the ratio for training data, 2nd element is the ratio for cross validation data. The values range from 0 to 1
		==========================================================================================
		Output ->
		return a tuple of 3 2-D arrays, one is training dataset, one is for cross-validation, and the third is for testing
		"""
		np.random.shuffle(data_extracted)
		num_rows = np.size(data_extracted[:,0])
		num_row_train = int(round(num_rows*split_ratio[0]))
		num_row_cross = int(round(num_rows*split_ratio[1]))
	#	num_row_test = num_rows - num_row_train - num_row_cross
		return (data_extracted[0:num_row_train,:], data_extracted[num_row_train:(num_row_train + num_row_cross),:], data_extracted[(num_row_train + num_row_cross):num_rows,:])
		print num_rows

	def two_clns(self, data_header, data_whole, clns, split_ratio = [0.6, 0.2, 0.2]):
		'''This method is to handle data stratification in two columns.

		Input ->
		data_header :   header/column names of the dataset
		data_whole :    2-D array of the whole dataset
		clns :          list of the names of 2 columns to be splitted by
		split_ratio :   list of ratios of train, cross validation and test datasets
		==============================================================================
		Output ->
		return a tuple of 3 2-D arrays of training, cross validation and test datasets
		===============================================================================
		Note ->
		This method is DEPRECIATED as the first 2 methods in this class is good enough to handle similar cases. This method is historical version.
		'''
	## Step 1. Find the index number of the columns according to their names	
		stratified_cln = [0,1]
		i = 0
		for cln_name in clns:
			stratified_cln[i] = np.where(data_header == cln_name)[1][0]
#!			print np.where(data_header == cln_name)
			i = i + 1
#!		print stratified_cln
	## Step 2. Create two lists, one is the list of lists of unique values of each of 2 clns, one is the list of # of unique values in each of 2 clns
		num_rows = np.size(data_whole[:,0])
		stratified_cln_unique_value = []
		temp_list = []
		for i in stratified_cln:
			temp_array = np.unique(data_whole[:,i])
			temp_list.append(list(temp_array))
			stratified_cln_unique_value.append(len(temp_array))
	#!	print temp_list[0][0]
	## Step 3. Segment the rows in to (m X n) segments, here m, n are # of unique values of in each of 2 clns
		data_train_array = data_header
		data_cross_array = data_header
		data_test_array = data_header
		for i in xrange(stratified_cln_unique_value[0]):
			for j in xrange(stratified_cln_unique_value[1]):
				data_extracted = data_whole[(data_whole[:,stratified_cln[0]] == temp_list[0][i]) & (data_whole[:,stratified_cln[1]] == temp_list[1][j]),:]
				(temp_train, temp_cross, temp_test) = self.data_randomnize(data_extracted, split_ratio)
	#!			print temp_train
				data_train_array = np.vstack((data_train_array, temp_train))
				data_cross_array = np.vstack((data_cross_array, temp_cross))
				data_test_array = np.vstack((data_test_array, temp_test))
	#!			print data_extracted
		return data_train_array, data_cross_array, data_test_array


class semantics(object):
	@classmethod
	def count_word_in_column(cls, word_list):
		"""
		word_list:  a list of words
		return: a set of all words occurred in the list, and a list of sets where each set contains the word
		"""
		word_set = set()
		list_of_word_set = [None] * len(word_list)
		for i, word in enumerate(word_list):
			tmp_set = set(word.split())
			list_of_word_set[i] = tmp_set
			word_set.update(tmp_set)
		return word_set, list_of_word_set

	@classmethod
	def count_word_in_column_advance(cls, word_list):
		"""This method uses advanced re module to split more dirty words string"""
		import re
		from nltk.stem.lancaster import LancasterStemmer
		word_set = set()
		list_of_word_set = [None] * len(word_list)
		for i, word in enumerate(word_list):
			tmp_lst = re.split('\/| +|\.|0|1|2|3|4|-|\(|\)|\'|400|7|@|\&', word) # Refer to: http://stackoverflow.com/questions/4998629/python-split-string-with-multiple-delimiters
			tmp_set = set()
			for ele in tmp_lst:# Iterate through all ele in the list and do stemming
				ele = LancasterStemmer().stem(ele)
				tmp_set.update((ele,))
				try:
					tmp_set.remove("")
				except KeyError:
					pass
				
				try:
					tmp_set.remove("\xed\xa2\xf3\xf1")
				except KeyError:
					pass

				try:
					tmp_set.remove("\xc3\xad\xc2\xa2\xc3\xb3\xc3\xb1")
				except KeyError:
					pass
			list_of_word_set[i] = tmp_set
			word_set.update(tmp_set)
#		word_set_II = set()
#		for word in word_set:
#			word = LancasterStemmer().stem(word)
#			word_set_II.update((word,))
#		try:
#			word_set.remove("")
#			word_set.remove("\xed\xa2\xf3\xf1")
#		except KeyError:
#			print "Values '' or '\xed\xa2\xf3\xf1' don't exists!"
		return word_set, list_of_word_set

	@classmethod
	def create_columns_on_word_count(cls, word_list, cln_name = "New_cln"):
		cln_name = cln_name + "_"
		word_set, list_of_word_set = cls.count_word_in_column_advance(word_list)
		row_len = len(word_list)
		col_len = len(word_set)
		cln_name_lst = [None] * col_len
		for i, word in enumerate(word_set):
			cln_name_lst[i] = cln_name + word
		df = pd.DataFrame(0, index = np.arange(row_len), columns = cln_name_lst)
		for i, set_of_word in enumerate(list_of_word_set):
			for word in set_of_word:
				try:
					df[cln_name+word][i] = 1
				except KeyError:
					print "The " + cln_name + "does not have existing data as the one provided by you" 
		return df


class FA(object):
	def __init__(self, X, n_components):
		self._X = X
		self._n_components = n_components
		length = len(self._n_components)
		self._pca_scores = [None]*length
		self._fa_scores = [None]*length


	def compute_scores(self):
		from sklearn.decomposition import PCA, FactorAnalysis
		from sklearn.cross_validation import cross_val_score
		pca = PCA()
		fa = FactorAnalysis()
		for i, n in enumerate(self._n_components):
			pca.n_components = n
			fa.n_components = n
			self._pca_scores[i] = np.mean(cross_val_score(pca, self._X))
			self._fa_scores[i] = np.mean(cross_val_score(fa,self._X))

	def plt_scores(self, title='Scores versus different components'):
		import matplotlib.pyplot as plt
		from sklearn.decomposition import PCA, FactorAnalysis
		n_components_pca = self._n_components[np.argmax(self._pca_scores)]
		n_components_fa = self._n_components[np.argmax(self._fa_scores)]

		pca = PCA(n_components='mle')
		pca.fit(self._X)
		n_components_pca_mle = pca.n_components_

		print("best n_components by PCA CV = %d" % n_components_pca)
		print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
		print("best n_components by PCA MLE = %d" % n_components_pca_mle)

		plt.figure()
		plt.plot(self._n_components, self._pca_scores, 'b', label='PCA scores')
		plt.plot(self._n_components, self._fa_scores, 'r', label='FA scores')
		plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
		plt.axvline(n_components_fa, color='r', label='FA CV: %d' % n_components_fa, linestyle='--')
		plt.axvline(n_components_pca_mle, color='k', label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

		plt.xlabel('no. of components')
		plt.ylabel('CV scores')
		plt.legend(loc='lower right')
		plt.title(title)
		plt.show()

class feat_plot(object):
	"""Collection of some customized plotting functions"""
	@classmethod
	def plot_feat_import(cls, clf, feat_names):
		"""Plot the importance of features"""
		import matplotlib.pyplot as plt
		feat_importance = clf.feature_importances_
		#make importances relative to max importance
		feat_importance = 100.0*(feat_importance / feat_importance.max())
		if len(feat_importance)>20:
			sorted_idx = np.argsort(feat_importance)[-20:]
		else:
			sorted_idx = np.argsort(feat_importance)
		pos = np.arange(sorted_idx.shape[0]) + 0.5
		plt.subplot(1,2,2)
		plt.barh(pos, feat_importance[sorted_idx], align="center")
		plt.yticks(pos, feat_names[sorted_idx])
		plt.xlabel("Relative Importance")
		plt.title("Variable Importance")
		plt.show()

class CorrelationCalculation(object):
	"""Collection of customized methods for calculate correlations"""
	@classmethod
	def cal_category_corr(cls, df):
		"""Given a data frame of categorical data with one-hot encoding, calculate the modified correlation"""
		feat_names = df.columns.values
		num_feat = len(feat_names) # Total number of features
		arr_new = np.zeros(num_feat, num_feat)
		X = df.values
		for i, feat in enumerate(feat_names):
			for j in np.arange(i, num_feat):
				arr_2_clns = np.vstack((X[:,i],X[:,j])).T # combine 2 targeted columns into 1 array
				arr_2_clns = arr_2_clns[~np.all(arr_2_clns == 0, axis = 1)]
				arr_new[i,j] = np.corrcoef(arr_2_clns.T)
				arr_new[j,i] = arr_new[i,j]


class recommendation(object):
	"""Contains methods used for creating challenges of recommendation system"""
	def noise_add(self, percentage): # This method is to add additional noises to the cross validation and test datasets, and those noise user-movie pairs won't be evaluated
		pass

class rakuten(slicing, sort_arr, lookup, read_write):
	"""Import Derived class first, and then the base class. Here loopup is derived class of read_write. Actually no need to inherit 'read_write'."""
	def function():
		pass

class mindef(data_clean, read_write, sort_arr, stratification, create_files_for_participant, evaluation_metric):
	"""Contains methods for data cleaning and splitting for MINDEF data challenge."""
	def function():
		pass

class mindef_benchmark(data_clean, read_write):
	"""Contains methods for cracking the MINDEF data challenge."""
	def function():
		pass