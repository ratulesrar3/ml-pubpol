# Helper functions for MLPP Diagnostic HW
# Obtains 311 requests for vacant/abandoned buildings, graffiti removals, and alley lights using chicago data portal API
# Augments request data with demographic info from ACS API


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
import json
from urllib.request import urlopen


def load(url, date_col='Creation Date', start='12/31/2016', end='01/01/2018'):
    '''
    Loads data from data portal url and returns filtered df
    '''
    start = datetime.strptime(start, '%m/%d/%Y')
    end = datetime.strptime(end, '%m/%d/%Y')
    df = pd.read_csv(url)
    df['date'] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    df = df[(df['date'] > start) & (df['date'] < end)]    
    return df


def drop_if(df, col_list=['Ward','ZIP Code','Police District','Community Area'], value=0.0):
    '''
    Remove rows with non meaningful values
    '''
    for col in col_list:
        df = df.drop(df.index[df[col] == value])        
    return df


def days_between(d1, d2):
    '''
    Compute number of days between request and completion
    '''
    d1 = datetime.strptime(d1, '%m/%d/%Y')
    d2 = datetime.strptime(d2, '%m/%d/%Y')
    return abs((d2 - d1).days)


def compute_response_time(df, col1='Creation Date', col2='Completion Date'):
    '''
    Compute response time for rows in df
    '''
    response_time_list = []
    df = df[~df[col2].isnull()]
    for start, end in zip(list(df[col1]), list(df[col2])):
        if type(start) and type(end) == str:
            days = days_between(end, start)
            response_time_list.append(days)
    df['Response Time (Days)'] = pd.Series(response_time_list, index=df.index)
    return df


def add_month_bins(df, col1='Creation Date'):
    '''
    Adds a numeric bin for the month that a request was created
    '''
    month_list = []
    for day in df[col1]:
        clean = datetime.strptime(day, '%m/%d/%Y')
        month_list.append(clean.month)
    df['Month'] = pd.Series(month_list, index=df.index)
    return df


def hist_plt(df, col, xlab, vertical=False, sort=True):
    '''
    Function to plot histograms
    '''
    plt.figure(figsize=(16,12))
    if sort:
        hist_idx = df[col].value_counts()
    else:
        hist_idx = df[col].value_counts(sort=False)
    if vertical:
            graph=sns.countplot(y=col, saturation=1, data=df, order=hist_idx.index)
            plt.ylabel(xlab)
            plt.xlabel('Num Requests')
    else:
        graph=sns.countplot(x=col, saturation=1, data=df, order=hist_idx.index)
        plt.xlabel(xlab)
        plt.ylabel('Num Requests')
    plt.title('Histogram Request Counts by '+col)
    plt.show()
    

def max_min_count(df, col1='Response Time (Days)', col2='ZIP Code'):
    '''
    Return the zip code with largest and lowest mean response times
    Return the zip code with the largest and lowest number of requests
    '''
    
    # zip code with the most requests
    a = df[col1].groupby(df[col2]).describe().unstack()['count'].idxmax()
    b = df[col1].groupby(df[col2]).describe().unstack()['count'].max()
    print(str(a) +' had ' + str(b) + ' requests, which was the most overall \n')
    
    # zip code with the fewest requests
    a = df[col1].groupby(df[col2]).describe().unstack()['count'].idxmin()
    b = df[col1].groupby(df[col2]).describe().unstack()['count'].min()
    print(str(a) +' had ' + str(b) + ' requests, which was the least overall \n')
    
    # zip code with highest mean response time
    a = df[col1].groupby(df[col2]).describe().unstack()['mean'].idxmax()
    b = df[col1].groupby(df[col2]).describe().unstack()['mean'].max()
    print(str(a) +' had an average response time of ' + str(b) + ' day, which was the highest overall \n')
    
    # zip code with the lowest mean response time
    a = df[col1].groupby(df[col2]).describe().unstack()['mean'].idxmin()
    b = df[col1].groupby(df[col2]).describe().unstack()['mean'].min()
    print(str(a) +' had an average response time of ' + str(b) + ' day, which was the lowest overall \n')
    
    # all zip codes
    print('Below is a summary of response times for all zip codes \n')
    return df[col1].groupby(df[col2]).describe()


def three_months(filename, date_col='Creation Date', start='09/30/2017', end='01/01/2018'):
    '''
    Loads data from pickle file and returns filtered df based on date
    '''
    df = pd.read_pickle(filename)
    start = datetime.strptime(start, '%m/%d/%Y')
    end = datetime.strptime(end, '%m/%d/%Y')
    df['date'] = pd.to_datetime(df[date_col], format='%m/%d/%Y')
    df = df[(df['date'] > start) & (df['date'] < end)]
    return df


def get_census_block(lat, long):
    '''
    Given a latitude and longitude, find the corresponding FIPS code
    '''
    FIPS_url = 'https://geo.fcc.gov/api/census/block/find?latitude={}&longitude={}&showall=true&format=json'.format(str(lat),str(long))
    try:
        response = urlopen(FIPS_url)
        FIPS = response.read().decode('utf-8')
        FIPS = json.loads(FIPS)
        return FIPS['Block']['FIPS']
    except:
        print(FIPS_url)

        
def scrape_fips_blocks(df, lat='Latitude', lon='Longitude'):
    '''
    Append fips block numbers to each request in a df
    '''
    blocks = []
    for index, row in df.iterrows():
        x = row[lat]
        y = row[lon]
        blocks.append(get_census_block(x, y))
    df['FIPS_BLOCK_NUMBER'] = pd.Series(blocks, index=df.index)
    return df


def scrape_acs(df):
    '''
    Function that retrieves the monthly income, family size, pct white or with health coverage, and unemployment
    Linkages found using a request's FIPS code
    Returns augemented df with demographic info if found
    '''
    avg_income_list = []
    avg_family_size = []
    pct_white = []
    health_coverage = []
    unemployment = []
    for index, row in df.iterrows():
        state = row['FIPS_BLOCK_NUMBER'][0:2]
        county = row['FIPS_BLOCK_NUMBER'][2:5]
        tract = row['FIPS_BLOCK_NUMBER'][5:11]
        query = 'get=NAME,DP03_0051E,DP02_0016E,DP05_0032PE,DP03_0096PE,DP03_0005PE&for=tract:{}&in=state:{}+county:{}&'.format(tract, state, county)
        key = 'key=02483eaf62001ddc247c04dc50cfa681d83bce62'
        url = 'https://api.census.gov/data/2015/acs5/profile?'+query+key
        r = requests.get(url)
        if r.status_code != 204: # 204 corresponds to no content.
            json = r.json()
            if json[1][1] is not '-':
                avg_income_list.append(int(json[1][1]))
            else:
                avg_income_list.append(np.nan)
            if json[1][2] is not '-':
                avg_family_size.append(float(json[1][2]))
            else:
                avg_family_size.append(np.nan)
            if json[1][3] is not '-':
                pct_white.append(float(json[1][3]))
            else:
                pct_white.append(np.nan)
            if json[1][4] is not '-':
                health_coverage.append(float(json[1][4]))
            else:
                health_coverage.append(np.nan)
            if json[1][5] is not '-':
                unemployment.append(float(json[1][5]))
            else:
                unemployment.append(np.nan)
    df['avg_monthly_income'] = pd.Series(avg_income_list, index=df.index)
    df['avg_family_size'] = pd.Series(avg_family_size, index=df.index)
    df['pct_white'] = pd.Series(pct_white, index=df.index)
    df['pct_health_coverage'] = pd.Series(health_coverage, index=df.index)
    df['unemployment_rate'] = pd.Series(unemployment, index=df.index)
    return df


def plot_dist(df, title, xlab):
    '''
    Given a df of a mean for each zip code, make a bar graph
    '''
    df.plot(kind='barh',figsize=(16, 12))
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel('Zip Codes')
    plt.show()
    
    
def combine_address(df, cols=['ADDRESS STREET NUMBER', 'ADDRESS STREET DIRECTION', 'ADDRESS STREET NAME', 'ADDRESS STREET SUFFIX']):
    '''
    Create Street Address column for buildings df
    '''
    address = []
    df[cols[0]] = df[cols[0]].astype('int')
    for index, row in df.iterrows():
        s = ''
        for i in range(len(cols)):
            if i is 0:
                s += str(row[cols[i]])
            else:
                s += ' ' + str(row[cols[i]])
        address.append(s)
    df['Street Address'] = pd.Series(address, index=df.index)
    return df


def search_by(df, col='Street Address', by='3600 W Roosevelt Ave'):
    '''
    Searches df[col] for term to return proportion of requests from that area
    '''
    df[col] = df[col].astype(str)
    df_filter = df[df[col].str.contains(by, case=False)]
    return len(df_filter) / len(df)


def combine_frames(files):
    '''
    Read data and combine the data into one table
    '''
    frames = [pd.read_pickle(file) for file in files]
    for frame in frames:
        frame.columns = map(str.lower, frame.columns)
    df = pd.concat(frames)

    # combine zip and zip code
    df['zip code'].fillna(df['zip code'], inplace=True)
    return df
    # drop other columns
    df.drop(df.columns[[0,1,2,3,4,9,10,11,12,13,14,17]], axis=1, inplace=True)
 
    # drop rows with NANs in essential columns
    df.dropna(subset=['community area','zip code','ward','police district','creation date','completion date','latitude'], inplace=True)

    # convert to categories
    cat = df[['type of service request','status','what type of surface is the graffiti on?']]
    df[['type of service request','status','what type of surface is the graffiti on?']] = cat.apply(lambda x: x.astype('category'))

    # convert to integer
    ints = df[['community area','police district','ward','zip code']]
    df[['community area','police district','ward','zip code']] = ints.apply(lambda x: x.astype('int'))

    # convert to datetime
    df['creation date'] = pd.to_datetime(df['creation date'],infer_datetime_format=True)
    df['completion date'] = pd.to_datetime(df['completion date'],infer_datetime_format=True)
    df['response time'] = ((df['completion date'] - df['creation date']) / np.timedelta64(1, 'D')).astype(int)

    return df