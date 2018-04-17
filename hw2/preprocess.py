# preprocessing and cleaning module
# ratul esrar, spring 18


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler


def make_bins(df, col, num_bins=15):
	'''
	Assigns column values into bins
	'''
	new_col = str(col)+'_bin'
	df[new_col] = pd.cut(df[col], bins=num_bins)

	return df


def categorical_dummies(df, columns):
	'''
	Pandas function wrapper to inplace combine a new set of dummy variable columns
	'''
	for column in columns:
		dummies = pd.get_dummies(df[column], prefix=column+"_is", prefix_sep='_', dummy_na=True)
		df = pd.concat([df, dummies], axis=1)

	return df


def impute_by(df, by='mean'):
	'''
	Replace the NaNs with the column mean, median, or mode
	'''
	null_cols = df.columns[pd.isnull(df).sum() > 0].tolist()
	for column in null_cols:
		# input the mean of the data if they are numeric
		data = df[column]
		if data.dtype in [int, float]:
			if by is 'median':
				imputed_value = data.median()
			elif by is 'mode':
				imputed_value = data.mode()
			else:
				imputed_value = data.mean()
			df.loc[:,(column)] = data.fillna(imputed_value)
	
	return df


def scale_cols(train_df, test_df, col):
    '''
    Scale columns that do not follow a well-defined distribution
    '''
    pd.options.mode.chained_assignment = None
    robust_scaler = RobustScaler()
    scaled_col = str(col)+'_scaled'
    train_df[scaled_col] = robust_scaler.fit_transform(train_df[col].values.reshape(-1,1))
    test_df[scaled_col] = robust_scaler.transform(test_df[col].values.reshape(-1,1))
    
    return train_df, test_df