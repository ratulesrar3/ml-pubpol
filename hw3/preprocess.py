# preprocessing and cleaning module
# ratul esrar, spring 18


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler


# columns to binarize
BINARY =['fully_funded', 'school_charter', 'school_magnet', 'school_year_round',
		 'school_nlns', 'school_kipp', 'school_charter_ready_promise',
		 'teacher_teach_for_america', 'teacher_ny_teaching_fellow',
         'eligible_double_your_impact_match', 'eligible_almost_home_match']

# colums to discretize
CATEGORICAL = ['school_metro', 'primary_focus_subject', 'primary_focus_area',
			   'secondary_focus_subject', 'secondary_focus_area',
			   'resource_type', 'grade_level', 'school_state', 'school_zip',
			   'teacher_prefix']

# columns to dummify
GEOGRAPHICAL = ['school_zip', 'school_disctrict']

# to incorporate later
LOCATION = ['school_latitude', 'school_longitude', 'school_city', 'school_state',
			'school_county', 'school_state', 'school_zip', 'school_district']

# columns to scale
SCALERS = ['fulfillment_labor_materials', 'total_price_excluding_optional_support',
           'total_price_including_optional_support', 'students_reached']

# columns to impute missing values
IMPUTE_BY = {'students_reached': 'mean'}

# date
DATE = ['date_posted']

# label
LABEL = ['fully_funded']

# to scale columns
SCALE = True

def impute_by(df, column, by='mean'):
	'''
	Replace the NaNs with the column mean, median, or mode
	'''
	if by == 'median':
	    df[column].fillna(df[column].median(), inplace=True)
	elif by == 'mode':
	    df[column].fillna(df[column].mode(), inplace=True)
	elif by == 'zero':
	    df[column].fillna(0, inplace=True)
	else:
		df[column].fillna(df[column].mean(), inplace=True)


def scale_cols(train, test, col):
    '''
    Scale columns that do not follow a well-defined distribution
    '''
    pd.options.mode.chained_assignment = None
    scaler = RobustScaler()
    scaled_col = str(col)+'_scaled'
    train[scaled_col] = scaler.fit_transform(train[col].values.reshape(-1,1))
    test[scaled_col] = scaler.transform(test[col].values.reshape(-1,1))

    return train[scaled_col], test[scaled_col]


def top_code_extrema(df, column, lb=0.001, ub=0.999):
    '''
    Top code extreme values based on the specifed quantile
    '''
    lb, ub = df[column].quantile(lb), df[column].quantile(ub)
    print('Column was capped between {} and {}.'.format(lb, ub))
    df[column] = df[column].apply(cap_value, args=(lb, ub))


def to_binary(df, col, val='t'):
    '''
    Converts column to binary based on value
    '''
    df[col] = df[col].apply(lambda x: 1 if x == val else 0)


def categorical_dummies(df, columns):
    '''
    Naive dummies from categorical vars
    '''
    for col in columns:
        #print(col)
        dummies = pd.get_dummies(df[col], prefix=col+"_is", prefix_sep='_', dummy_na=False)
        df = pd.concat([df, dummies], axis=1)

    df = df.drop(columns, axis=1)


def top_n_categories(df, col, n=5):
    '''
    Get top n categories in a column and nan/others
    '''
    dummies = set(df[col].value_counts().head(n).index)
    dummies = dummies.union(set([np.nan, 'others']))
    return dummies


def to_discrete(dummies, df, col):
    for val in dummies:
        col_name = '{}_is_{}'.format(col, str(val))
        if val != 'others':
            df[col_name] = df[col].apply(lambda x: 1 if x == val else 0)
        else:
            df[col_name] = df[col].apply(lambda x: 1 if x not in dummies else 0)
        df = pd.concat([df, df[col_name]], axis=1)


def pre_process(train, test):
	'''
	Processes train and test dataframes to prep for loop
	'''
	features = set()
	for col in train.columns:
		if col in BINARY:
			#print('Binarized: {}'.format(col))
			to_binary(train, col, 't')
			to_binary(test, col, 't')
			features.add(col)
		if col in CATEGORICAL:
			#print('Discretized: {}'.format(col))
			dummies = top_n_categories(train, col, 5)
			x = set(['{}_is_{}'.format(col, str(val)) for val in dummies])
			#print(x)
			features = features.union(x)
			to_discrete(dummies, train, col)
			to_discrete(dummies, test, col)
		if col in GEOGRAPHICAL:
			#print('Dummified: {}'.format(col))
			features.add(col)
			categorical_dummies(train, [col])
			categorical_dummies(test, [col])
		if col in IMPUTE_BY:
			features.add(col)
			#print('Imputed: {}'.format(col))
			impute_by(train, col, IMPUTE_BY[col])
			impute_by(test, col, IMPUTE_BY[col])
		if SCALE and col in SCALERS:
			#print('Scaled: {}'.format(col))
			features.add(col)
			train[col], test[col] = scale_cols(train, test, col)
	return train, test, features
