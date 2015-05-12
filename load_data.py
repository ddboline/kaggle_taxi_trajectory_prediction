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

import gzip
import csv
import time

#from feature_extraction import haversine_distance
#from feature_extraction import LATLIM, LONLIM
from feature_extraction import get_trajectory, get_matching_list

from feature_extraction import haversine_distance, compare_trajectories
#try:
#    import pyximport
#    pyximport.install()
#    from compare_trajectories import haversine_distance, compare_trajectories
#except ImportError:
#    from feature_extraction import haversine_distance, compare_trajectories

def clean_data(df):
    df['CALL_TYPE'] = df['CALL_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df['DAY_TYPE'] = df['DAY_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df['WEEK'] = df['TIMESTAMP'].apply(lambda x: x % (7*24*3600))
    df['DAY'] = df['TIMESTAMP'].apply(lambda x: x % (24*3600))
    df['HOUR'] = df['TIMESTAMP'].apply(lambda x: x % (3600))
    df.loc[(df['ORIGIN_CALL'].isnull()), 'ORIGIN_CALL'] = -1
    df.loc[(df['ORIGIN_STAND'].isnull()), 'ORIGIN_STAND'] = -1
    for col in ('ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA',
                'TRAJECTORY_IDX'):
        df[col] = df[col].astype(int)
    df = df.dropna(axis=0, subset=['ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT',
                                   'DEST_LON'])

    df = df.drop(labels=['DAY_TYPE', 'TRIP_ID'], axis=1)
    return df
    
def find_best_traj(do_plots=False):
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

    train_nib = pd.read_csv('train_nib.csv.gz', compression='gzip')
    test_nib = pd.read_csv('test_nib.csv.gz', compression='gzip')

    test_trj = pd.read_csv('test_trj.csv.gz', compression='gzip')

    randperm = np.random.permutation(np.arange(train_df.shape[0]))
    dfs = [{'df': test_df, 'fn': 'test_final.csv.gz', 'test': True},
           {'df': train_df.iloc[randperm[:320], :],
            'fn': 'train_final.csv.gz', 'test': False},
           {'df': train_df.iloc[randperm[320:640], :],
            'fn': 'valid_final.csv.gz', 'test': False}]
    
    outlabels = ['CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID',
                 'TIMESTAMP', 'BEST_LAT', 'BEST_LON', 'AVG_LAT', 'AVG_LON',
                 'DEST_LAT', 'DEST_LON']

    for dfs_dict in dfs:
        df = dfs_dict['df']
        outfname = dfs_dict['fn']
        is_test = dfs_dict['test']
        outfile = gzip.open(outfname, 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(outlabels)
        print(outfname)
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print('test %d' % idx)
            tidx = row['TRAJECTORY_IDX']
            if is_test:
                tdf_ = test_trj
            else:
                tdf_ = pd.read_csv('train/train_trj_%02d.csv.gz' % tidx,
                                   compression='gzip')
            traj_ = get_trajectory(tidx, train_df=tdf_)
            if is_test:
                tedf_ = test_nib
            else:
                tedf_ = train_nib
            
            def get_common_trajectories(mindist=0.05, rebin=1):
                match_list_ = get_matching_list(tidx, test_df=tedf_,
                                                train_df=train_nib)
                common_traj = {}
                time_0 = time.clock()
                for fidx in range(100):
                    if fidx % 10 == 0:
                        print('fidx %d' % fidx)
                    train_trj_ = pd.read_csv('train/train_trj_%02d.csv.gz'
                                             % fidx, compression='gzip')
                    n_matching = 0
                    for idx_, tidx in enumerate(match_list_):
                        if tidx % 100 != fidx:
                            continue
                        if tidx in randperm[:640]:
                            continue
                        n_matching += 1
                        train_traj_ = get_trajectory(tidx, train_df=train_trj_)
                        n_common = compare_trajectories(traj_, train_traj_)
                        if n_common == 0:
                            continue
                        common_traj[tidx] = n_common
                    time_1 = time.clock()
                    print('time %s %s %s' % (time_1-time_0, len(common_traj),
                                             n_matching))
                    time_0 = time_1
                return common_traj
            common_traj = get_common_trajectories()
            if len(common_traj) == 0:
                common_traj = get_common_trajectories(0.1, 10)
            sort_list = sorted(common_traj.items(), key=lambda x: x[1])
            cond = train_df['TRAJECTORY_IDX'] == sort_list[-1][0]
            best_lat = float(train_df[cond]['DEST_LAT'])
            best_lon = float(train_df[cond]['DEST_LON'])
            top_lats = []
            top_lons = []
            for k, v in sort_list[-10:]:
                cond = train_df['TRAJECTORY_IDX'] == k
                top_lats.append(float(train_df[cond]['DEST_LAT']))
                top_lons.append(float(train_df[cond]['DEST_LON']))
            avg_lat = np.mean(top_lats)
            avg_lon = np.mean(top_lons)
            dist = haversine_distance(best_lat, best_lon, avg_lat, avg_lon)
            print('%s' % dist)
            row_dict = dict(row)
            row_dict['BEST_LAT'] = best_lat
            row_dict['BEST_LON'] = best_lon
            row_dict['AVG_LAT'] = avg_lat
            row_dict['AVG_LON'] = avg_lon
            for k in row_dict:
                if k in ('ORIGIN_LAT', 'ORIGIN_LON', 'TOTAL_DISTANCE',
                         'BEST_LAT', 'BEST_LON', 'AVG_LAT', 'AVG_LON',
                         'DEST_LAT', 'DEST_LON'):
                    continue
                row_dict[k] = int(row_dict[k])
            row_val = [row_dict[k] for k in outlabels]
            csv_writer.writerow(row_val)
#            if idx == 1:
#                exit(0)
    return

if __name__ == '__main__':
    find_best_traj(do_plots=False)
