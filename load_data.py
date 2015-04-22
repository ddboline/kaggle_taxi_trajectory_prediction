#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:40:52 2015

@author: ddboline
"""
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    r_earth = 6371.
    dlat = np.abs(lat1-lat2)*np.pi/180.
    dlon = np.abs(lon1-lon2)*np.pi/180.
    lat1 *= np.pi/180.
    lat2 *= np.pi/180.
    dist = 2. * r_earth * np.arcsin(
                            np.sqrt(
                                np.sin(dlat/2.)**2 + 
                                    np.cos(lat1) * np.cos(lat2) * 
                                    np.sin(dlon/2.)**2))
    return dist

def clean_data(df):
    df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df['DAY_TYPE'] = df['DAY_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df.loc[(df['ORIGIN_CALL'].isnull()), 'ORIGIN_CALL'] = -1
    df.loc[(df['ORIGIN_STAND'].isnull()), 'ORIGIN_STAND'] = -1
    df['MISSING_DATA'] = df['MISSING_DATA'].astype(int)
    return df
    
def load_data():
    train_df = pd.read_csv('train.csv.gz', compression='gzip', nrows=1000)
    
    train_df = clean_data(train_df)    
    
    print train_df.columns
    print train_df.dtypes
    print sorted(train_df['ORIGIN_CALL'].unique())
    print sorted(train_df['ORIGIN_STAND'].unique())
#    print train_df['POLYLINE'].head()
   
    return

if __name__ == '__main__':
    load_data()