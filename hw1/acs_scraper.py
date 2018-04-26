### helper functions for obtaining infromation from the ACS census API
### Ratul Esrar, Winter 2018


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
import json
from urllib.request import urlopen


def get_census_block(lat, long):
    '''
    Given a latitude and longitude, find the corresponding FIPS code using the fcc api
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
    Append fips block numbers to each row in a dataframe using the row's lat-lon
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
    Function that retrieves the monthly income, family size, pct white, pct with health coverage, and unemployment
    Linkages found using a row's FIPS code
    Returns augemented dataframe with demographic info if found

    dictionary of columns being scraped:
    {'DP03_0051E': 'income in 1000s',
    'DP02_0016E': 'average family size',
    'DP05_0032PE': 'percent white',
    'DP03_0096PE': 'percent with health coverage',
    'DP03_0005PE': 'percent unemployed'}
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

        # the following chunk checks if the api returns a value for the given census tract
        # appends NaN if nothing found
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