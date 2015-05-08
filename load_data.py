#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 14:40:52 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pandas as pd

#from feature_extraction import haversine_distance
#from feature_extraction import LATLIM, LONLIM
from feature_extraction import get_lat_bin, get_lon_bin, get_trajectory

def clean_data(df):
    df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df['DAY_TYPE'] = df['DAY_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df.loc[(df['ORIGIN_CALL'].isnull()), 'ORIGIN_CALL'] = -1
    df.loc[(df['ORIGIN_STAND'].isnull()), 'ORIGIN_STAND'] = -1
    for col in ('ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA'):
        df[col] = df[col].astype(int)
    df['ORIGIN_LATBIN'] = df['ORIGIN_LAT'].apply(get_lat_bin)
    df['ORIGIN_LONBIN'] = df['ORIGIN_LON'].apply(get_lon_bin)
    df['DEST_LATBIN'] = df['DEST_LAT'].apply(get_lat_bin)
    df['DEST_LONBIN'] = df['DEST_LON'].apply(get_lon_bin)
    
    df = df.drop(labels=['DAY_TYPE', 'TRIP_ID'], axis=1)
    return df
    
def load_data(do_plots=False):
    train_df = pd.read_csv('train_idx.csv.gz', compression='gzip')
    test_df = pd.read_csv('test_idx.csv.gz', compression='gzip')
    
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    print(train_df.shape, test_df.shape)
    print(test_df.dtypes)

    if do_plots:
        from plot_data import plot_data
        plot_data(train_df, prefix='train_html', do_scatter=False)
        plot_data(test_df, prefix='test_html', do_scatter=False)
    
    tjidx = train_df.loc[0, 'TRAJECTORY_IDX']
    print(get_trajectory(trj_idx=tjidx))
    return

if __name__ == '__main__':
    load_data(do_plots=False)
