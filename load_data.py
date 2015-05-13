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

from feature_extraction import get_trajectory, get_matching_list

from feature_extraction import haversine_distance, compare_trajectories

import multiprocessing

def find_common_trajectories(args):
    traj_, fidx, match_list_, skiplist = args
    print(len(match_list_), len(skiplist))
    time_0 = time.clock()
    train_trj_ = pd.read_csv('train/train_trj_%02d.csv.gz'
                             % fidx, compression='gzip')
    n_matching = 0
    n_matched = 0
    common_traj = {}
    for tidx in match_list_:
        if tidx % 100 != fidx:
            continue
        if tidx in skiplist:
            continue
        n_matching += 1
        train_traj_ = get_trajectory(tidx, tr_df=train_trj_)
        n_common = compare_trajectories(traj_, train_traj_, mindist=0.1)
        if n_common == 0:
            continue
        common_traj[tidx] = n_common
        n_matched += 1
    time_1 = time.clock()
    print('time %s %s %s' % (time_1-time_0, n_matched,
                             n_matching))
    return common_traj

def clean_data(df_):
    """
        Clean and update DataFrames
    """
    df_['CALL_TYPE'] = df_['CALL_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df_['DAY_TYPE'] = df_['DAY_TYPE'].map({'A': 0, 'B': 1, 'C': 2})
    df_['WEEK'] = df_['TIMESTAMP'].apply(lambda x: x % (7*24*3600))
    df_['DAY'] = df_['TIMESTAMP'].apply(lambda x: x % (24*3600))
    df_['HOUR'] = df_['TIMESTAMP'].apply(lambda x: x % (3600))
    df_.loc[(df_['ORIGIN_CALL'].isnull()), 'ORIGIN_CALL'] = -1
    df_.loc[(df_['ORIGIN_STAND'].isnull()), 'ORIGIN_STAND'] = -1
    for col in ('ORIGIN_CALL', 'ORIGIN_STAND', 'MISSING_DATA',
                'TRAJECTORY_IDX'):
        df_[col] = df_[col].astype(int)
    df_ = df_.dropna(axis=0, subset=['ORIGIN_LAT', 'ORIGIN_LON', 'DEST_LAT',
                                   'DEST_LON'])

#    df_ = df_.drop(labels=['DAY_TYPE', 'TRIP_ID'], axis=1)
    return df_

def find_best_traj(do_plots=False):
    """
        Find the best trajectories from "template" sample
    """
    ncpu = len(filter(lambda x: x.find('processor') == 0,
                      open('/proc/cpuinfo')
                      .read().split('\n')))
    print('ncpu', ncpu)

    pool = multiprocessing.Pool(ncpu*2)

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

    outlabels = ['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND',
                 'TAXI_ID', 'TIMESTAMP', 'BEST_LAT', 'BEST_LON', 'AVG_LAT',
                 'AVG_LON', 'DEST_LAT', 'DEST_LON']

    for dfs_dict in dfs:
        df_ = dfs_dict['df']
        outfname = dfs_dict['fn']
        is_test = dfs_dict['test']
        outfile = gzip.open(outfname, 'wb')
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(outlabels)
        print(outfname)
        for idx, row in df_.iterrows():
            if idx % 10 == 0:
                print('test %d' % idx)
            tidx = row['TRAJECTORY_IDX']
            if is_test:
                tdf_ = test_trj
            else:
                tdf_ = pd.read_csv('train/train_trj_%02d.csv.gz' % tidx,
                                   compression='gzip')
            traj_ = get_trajectory(tidx, tr_df=tdf_)
            if is_test:
                tedf_ = test_nib
            else:
                tedf_ = train_nib
            common_traj = {}
            match_list_ = get_matching_list(tidx, te_df=tedf_,
                                            tr_df=train_nib)
            match_list_parallel = [{} for i in range(100)]
            for tidx in match_list_:
                match_list_parallel[tidx%100][tidx] = match_list_[tidx]
            skiplist = tuple(randperm[:640])
            parallel_args = [(traj_, i, match_list_parallel[i], skiplist)
                             for i in range(100)]
            for out_traj_ in pool.imap_unordered(find_common_trajectories,
                                                 parallel_args):
                for k, v in out_traj_.items():
                    common_traj[k] = v
            sort_list = sorted(common_traj.items(), key=lambda x: x[1])
            cond = train_df['TRAJECTORY_IDX'] == sort_list[-1][0]
            best_lat = float(train_df[cond]['DEST_LAT'])
            best_lon = float(train_df[cond]['DEST_LON'])
            top_lats = []
            top_lons = []
            for key, _ in sort_list[-10:]:
                cond = train_df['TRAJECTORY_IDX'] == key
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
