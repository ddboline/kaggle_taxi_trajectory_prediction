#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:00 2015

@author: ddboline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import gzip
import csv
from math import sqrt, sin, cos, pi, asin

import numpy as np

from collections import defaultdict

LATLIM = (41.1, 41.2)
LONLIM = (-8.7, -8.5)

NBINS = 1000

def get_lat_bin(lat, nbins=NBINS):
    """ get latitude bin """
    if lat < LATLIM[0]:
        return 0
    if lat > LATLIM[1]:
        return nbins
    return int((lat-LATLIM[0]) * nbins / (LATLIM[1]-LATLIM[0]))

def get_lon_bin(lon, nbins=NBINS):
    """ get longitude bin """
    if lon < LONLIM[0]:
        return 0
    if lon > LONLIM[1]:
        return nbins
    return int((lon-LONLIM[0]) * nbins / (LONLIM[1]-LONLIM[0]))

def haversine_distance(lat1, lon1, lat2, lon2):
    """ get Haversine Distance """
    r_earth = 6371.
    dlat = abs(lat1-lat2)*pi/180.
    dlon = abs(lon1-lon2)*pi/180.
    lat1 *= pi/180.
    lat2 *= pi/180.
    dist = 2. * r_earth * asin(sqrt(sin(dlat/2.)**2 +
                                      cos(lat1)*cos(lat2)*sin(dlon/2.)**2))
    return dist

def lat_lon_box(lat, dist):
    """ find lat/lon box of size dist*2 """
    r_earth = 6371.
    d_2r = dist/(2.*r_earth)
    dlat = 2. * (d_2r)
    dlon = 2. * np.arcsin((np.sin(d_2r))/(np.cos(lat)))
    dlat *= 180./np.pi
    dlon *= 180./np.pi
    return abs(dlat), abs(dlon)

def split_polyline(polyline_str):
    """ Split POLYLINE string """
    latlons = []
    missing = 0
    for idx, latlon in enumerate(polyline_str.split('],[')):
        latlon = latlon.replace('[[', '').replace(']]', '')
        if latlon == '[]':
            continue
        lon, lat = [float(x) for x in latlon.split(',')]

        dis = 0
        if len(latlons) > 0:
            dis = haversine_distance(lat, lon, latlons[-1][0], latlons[-1][1])

        if dis > 1.0 * (1 + missing):
            missing += 1
            continue

        if lat < LATLIM[0] or lat > LATLIM[1] or lon < LONLIM[0] or \
                lon > LONLIM[1]:
            if idx == 0:
                missing += 1
                continue

        latlons.append((lat, lon, dis))
    return latlons

def feature_extraction(is_test=False):
    """ extract features """
    taxi_stand_latlon = {}
    with gzip.open('metaData_taxistandsID_name_GPSlocation.csv.gz', 'rb') as \
            mfile:
        csv_reader = csv.reader(mfile)
        labels = next(csv_reader)
        for idx, row in enumerate(csv_reader):
            row_dict = dict(zip(labels, row))
            taxi_stand_latlon[int(row_dict['ID'])] = {
                'DESC': row_dict[u'Descricao'],
                'LAT': row_dict['Latitude'],
                'LON': row_dict['Longitude']}

    if is_test:
        output_file_idx = gzip.open('test_idx.csv.gz', 'wb')
        output_file_trj = [gzip.open('test_trj.csv.gz', 'wb')]
        output_file_nib = gzip.open('test_nib.csv.gz', 'wb')
    else:
        if not os.path.exists('train'):
            os.makedirs('train')
        output_file_idx = gzip.open('train_idx.csv.gz', 'wb')
        output_file_trj = [
            gzip.open('train/train_trj_%02d.csv.gz' % idx, 'wb')
            for idx in range(100)]
        output_file_nib = gzip.open('train_nib.csv.gz', 'wb')

    csv_writer_idx = csv.writer(output_file_idx)
    csv_writer_trj = [csv.writer(f) for f in output_file_trj]
    csv_writer_nib = csv.writer(output_file_nib)
    n_trj_file = len(csv_writer_trj)

    input_file = 'train.csv.gz'
    if is_test:
        input_file = 'test.csv.gz'

    with gzip.open(input_file, 'rb') as infile:
        csv_reader = csv.reader(infile)
        labels = next(csv_reader)
        new_labels_idx = ['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL',
                          'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE',
                          'MISSING_DATA', 'TRAJECTORY_IDX', 'ORIGIN_LAT',
                          'ORIGIN_LON', 'NUMBER_POINTS', 'TOTAL_DISTANCE',
                          'DEST_LAT', 'DEST_LON']
        new_labels_trj = ['TRAJECTORY_IDX', 'POINT_IDX', 'LAT', 'LON', 'LATBIN',
                          'LONBIN']
        csv_writer_idx.writerow(new_labels_idx)
        for csvf in csv_writer_trj:
            csvf.writerow(new_labels_trj)
        csv_writer_nib.writerow(['TRAJECTORY_IDX', 'LATBIN', 'LONBIN'])
        for idx, row in enumerate(csv_reader):
            row_dict = dict(zip(labels, row))
            latlon_points = split_polyline(row_dict['POLYLINE'])
            tot_dist = 0
            bin_set = set()
            for idy, lat_lon in enumerate(latlon_points):
                lat, lon, dis = lat_lon
                lat_bin = get_lat_bin(lat)
                lon_bin = get_lon_bin(lon)
                bin_set.add((lat_bin, lon_bin))
                tot_dist += dis
                row_val = [idx, idy, lat, lon, lat_bin, lon_bin]
                csv_writer_trj[idx % n_trj_file].writerow(row_val)
            for latb, lonb in sorted(bin_set):
                csv_writer_nib.writerow([idx, latb, lonb])

            n_points = len(latlon_points)
            if n_points == 0:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = ('nan', 'nan')
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = ('nan', 'nan')
            elif n_points == 1:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = \
                    latlon_points[0][:2]
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = \
                    latlon_points[0][:2]
            elif n_points == 2:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = \
                    latlon_points[0][:2]
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = \
                    latlon_points[1][:2]
            else:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = \
                    latlon_points[0][:2]
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = \
                    latlon_points[-1][:2]

            if row_dict['ORIGIN_CALL'] == 'NA':
                row_dict['ORIGIN_CALL'] = ''
            if row_dict['ORIGIN_STAND'] == 'NA':
                row_dict['ORIGIN_STAND'] = ''
            if row_dict['ORIGIN_STAND'] != "":
                ost = int(row_dict['ORIGIN_STAND'])
                if row_dict['ORIGIN_LAT'] == 'nan' \
                        and row_dict['ORIGIN_LON'] == 'nan':
                    row_dict['ORIGIN_LAT'] = taxi_stand_latlon[ost]['LAT']
                    row_dict['ORIGIN_LON'] = taxi_stand_latlon[ost]['LON']

            row_dict['TRAJECTORY_IDX'] = idx

            row_dict['NUMBER_POINTS'] = n_points
            row_dict['TOTAL_DISTANCE'] = tot_dist

            row_val = [row_dict[col] for col in new_labels_idx]
            csv_writer_idx.writerow(row_val)
            if idx % 10000 == 0:
                print('processed %d' % idx)
#            if idx > 10000:
#                exit(0)

    output_file_idx.close()
    output_file_nib.close()
    for outf in output_file_trj:
        outf.close()
    return

def get_trajectory(trj_idx=None, train_df=None):
    """ Get trajectory """
    return train_df[train_df['TRAJECTORY_IDX'] == trj_idx]\
                         [['LAT', 'LON']].values

def compare_trajectories(test_trj, train_trj, mindist=0.05):
    """ Compare two Trajectories"""
    n_common = 0
    for test_lat, test_lon in test_trj:
        dlat, dlon = lat_lon_box(test_lat, mindist*2)
        n_common_tr = 0
        for train_lat, train_lon in train_trj:
            print(train_lat, test_lat, train_lon, test_lon)
            if abs(train_lat-test_lat) > dlat or \
                    abs(train_lon-test_lon) > dlon:
                continue
            dis = haversine_distance(test_lat, test_lon, train_lat, train_lon)
            if dis < mindist:
                n_common_tr += 1
        if n_common_tr > 0:
            n_common += 1
    return n_common

def get_matching_list(tidx=None, test_df=None, train_df=None, rebinning=1):
    """ Get list of matching Trajectories """
    latlon_list = set()
    matching_list = defaultdict(int)
    for _, row in test_df[test_df['TRAJECTORY_IDX'] == tidx].iterrows():
        latlon_list.add((row['LATBIN']//rebinning, row['LONBIN']//rebinning))

    for latbin, lonbin in latlon_list:
        cond0 = (train_df['LATBIN']//rebinning) == latbin
        cond1 = (train_df['LONBIN']//rebinning) == lonbin
        trj_arr = sorted(train_df[cond0 & cond1]['TRAJECTORY_IDX'].unique())
        print(trj_arr)
        for tidx in trj_arr:
            matching_list[tidx] += 1
    return matching_list

if __name__ == '__main__':
    feature_extraction(is_test=False)
    feature_extraction(is_test=True)
