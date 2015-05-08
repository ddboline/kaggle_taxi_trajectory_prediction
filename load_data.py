#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:40:52 2015

@author: ddboline
"""
import numpy as np
import pandas as pd

#from feature_extraction import haversine_distance

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