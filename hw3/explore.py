# loading and exploring module
# ratul esrar, spring 18


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split


def read_from_csv(filename, index=None, split=False, target=None):
	'''
	Using os.path.splitext(file_name)[-1].lower(), find the extension of filename and then read into pandas dataframe
	Found here for reference:
	    http://stackoverflow.com/questions/5899497/checking-file-extension
	'''
	ext = os.path.splitext(filename)[-1].lower()

	if ext == '.csv':
		df = pd.read_csv(filename, index_col=index)
		if split:
			X_train, X_test, y_train, y_test = train_test_split(df.drop([target], axis=1), df[target], test_size=0.25, random_state=3)
			return X_train, X_test, y_train, y_test
	else:
		print('Incorrect file extension')
	return df


def count_nulls(df):
	'''
	Return number of null values for each column
	'''
	print('Null Values:')

	nulls = []
	for col in df.columns:
	    if df[col].isnull().any():
	        null_count = df[col].isnull().sum()
	        print(col, null_count, df[col].dtype)
	        nulls.append(col)
	return nulls


def plot_correlations(df, title):
	'''
	Plot heatmap of columns in dataframe
	'''
	ax = plt.axes()
	corr = df.corr()
	sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, ax=ax)
	ax.set_title(title)


def plot_dist(df, col, title, normal=True):
	'''
	Plot distribution of a column
	'''
	ax = plt.axes()
	if normal:
		sns.distplot(df[col], fit=stats.norm, kde=False, ax=ax)
	else:
		sns.distplot(df[col], kde=False, ax=ax)
	ax.set_title(title)
