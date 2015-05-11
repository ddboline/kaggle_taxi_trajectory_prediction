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

import numpy as np
import pandas as pd

#from feature_extraction import haversine_distance
#from feature_extraction import LATLIM, LONLIM
from feature_extraction import get_trajectory, get_matching_list, \
                               compare_trajectories

def clean_data(df):
    df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df['DAY_TYPE'] = df['DAY_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df.loc[(df['ORIGIN_CALL'].isnull()), 'ORIGIN_CALL'] = -1
    df.loc[(df['ORIGIN_STAND'].isnull()), 'ORIGIN_STAND'] = -1
    for col in ('ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA',
                'TRAJECTORY_IDX'):
        df[col] = df[col].astype(int)
    df = df.dropna(axis=0, subset=['ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT',
                                   'DEST_LON'])

    df = df.drop(labels=['DAY_TYPE', 'TRIP_ID'], axis=1)
    return df
    
def load_data(do_plots=False):
    train_df = pd.read_csv('train_idx.csv.gz', compression='gzip')
    test_df = pd.read_csv('test_idx.csv.gz', compression='gzip')
    submit_df = pd.read_csv('sampleSubmission.csv.gz', compression='gzip')
    
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)
    
    print(train_df.shape, test_df.shape, submit_df.shape)
    print(test_df.dtypes)

    if do_plots:
        from plot_data import plot_data
        plot_data(train_df, prefix='train_html', do_scatter=False)
        plot_data(test_df, prefix='test_html', do_scatter=False)

    for idx, row in test_df.iterrows():
        tidx = row['TRAJECTORY_IDX']
        traj_ = get_trajectory(tidx, is_test=True)
        print(traj_)
        match_list_ = get_matching_list(tidx, is_test=True)
        n_good_traj = 0
        for idx_, tidx in enumerate(match_list_):
            if idx_ % 10 == 0:
                print('idx_ %d' % idx_)
            train_traj_ = get_trajectory(tidx)
            n_common = compare_trajectories(traj_, train_traj_)
            if n_common == 0:
                continue
            print('n_common', n_common)
            n_good_traj += 1
        if idx == 0:
            print('n_good_traj', n_good_traj)
            exit(0)
    return

if __name__ == '__main__':
    load_data(do_plots=False)
