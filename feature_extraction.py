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

from collections import defaultdict

LATLIM = (41.1, 41.2)
LONLIM = (-8.7, -8.5)

def get_lat_bin(lat, nbins=100):
    if lat < LATLIM[0]:
        return 0
    if lat > LATLIM[1]:
        return nbins
    return int((lat-LATLIM[0]) * nbins / (LATLIM[1]-LATLIM[0]))

def get_lon_bin(lon, nbins=100):
    if lon < LONLIM[0]:
        return 0
    if lon > LONLIM[1]:
        return nbins
    return int((lon-LONLIM[0]) * nbins / (LONLIM[1]-LONLIM[0]))

def haversine_distance(lat1, lon1, lat2, lon2):
    r_earth = 6371.
    dlat = abs(lat1-lat2)*pi/180.
    dlon = abs(lon1-lon2)*pi/180.
    lat1 *= pi/180.
    lat2 *= pi/180.
    dist = 2. * r_earth * asin(sqrt(sin(dlat/2.)**2 +
                                      cos(lat1)*cos(lat2)*sin(dlon/2.)**2))
    return dist

def split_polyline(polyline_str):
    latlons = []
    missing = 0
    for idx, latlon in enumerate(polyline_str.split('],[')):
        latlon = latlon.replace('[[', '').replace(']]', '')
        if latlon == '[]':
            continue
        try:
            lon, lat = map(float, latlon.split(','))
        except:
            print(type(latlon), latlon)
            exit(0)

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
        if not os.path.exists('test'):
            os.makedirs('test')
        output_file_idx = gzip.open('test_idx.csv.gz', 'wb')
        output_file_trj = [gzip.open('test_trj.csv.gz', 'wb')]
        output_file_bin = [[
            gzip.open('test/test_bin_%02d_%02d.csv.gz' % (idx, jdx), 'wb')
            for jdx in range(11)] for idx in range(11)]
        output_file_nib = [gzip.open('test_nib.csv.gz', 'wb')]
    else:
        if not os.path.exists('train'):
            os.makedirs('train')
        output_file_idx = gzip.open('train_idx.csv.gz', 'wb')
        output_file_trj = [
            gzip.open('train/train_trj_%02d.csv.gz' % idx, 'wb')
            for idx in range(100)]
        output_file_bin = [[
            gzip.open('train/train_bin_%02d_%02d.csv.gz' % (idx, jdx), 'wb')
            for jdx in range(11)] for idx in range(11)]
        output_file_nib = [
            gzip.open('train/train_nib_%02d.csv.gz' % idx, 'wb')
            for idx in range(100)]

    csv_writer_idx = csv.writer(output_file_idx)
    csv_writer_trj = [csv.writer(f) for f in output_file_trj]
    csv_writer_nib = [csv.writer(f) for f in output_file_nib]
    n_trj_file = len(csv_writer_trj)


    input_file = 'train.csv.gz'
    if is_test:
        input_file = 'test.csv.gz'

    latlim = [None, None]
    lonlim = [None, None]
    with gzip.open(input_file, 'rb') as infile:
        csv_reader = csv.reader(infile)
        labels = next(csv_reader)
        new_labels_idx = ['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL',
                          'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE',
                          'MISSING_DATA', 'TRAJECTORY_IDX', 'ORIGIN_LAT',
                          'ORIGIN_LON', 'NUMBER_POINTS', 'TOTAL_DISTANCE',
                          'DEST_LAT', 'DEST_LON']
        new_labels_trj = ['TRAJECTORY_IDX', 'POINT_IDX', 'LAT', 'LON']
        csv_writer_idx.writerow(new_labels_idx)
        for csvf in csv_writer_trj:
            csvf.writerow(new_labels_trj)
        for csvf in csv_writer_nib:
            csvf.writerow(['TRAJECTORY_IDX', 'LATBIN', 'LONBIN'])
        for idx, row in enumerate(csv_reader):
            row_dict = dict(zip(labels, row))
            latlon_points = split_polyline(row_dict['POLYLINE'])
            tot_dist = 0
            bin_set = set()
            for idy, lat_lon in enumerate(latlon_points):
                lat, lon, dis = lat_lon
                if latlim[0] is None or latlim[0] > lat:
                    latlim[0] = lat
                if latlim[1] is None or latlim[1] < lat:
                    latlim[1] = lat
                if lonlim[0] is None or lonlim[0] > lon:
                    lonlim[0] = lon
                if lonlim[1] is None or lonlim[1] < lon:
                    lonlim[1] = lon
                latgrid = int((lat-LATLIM[0]) * 100 / (LATLIM[1]-LATLIM[0]))
                if latgrid < 0:
                    latgrid = 0
                if latgrid >= 100:
                    latgrid = 100
                longrid = int((lon-LONLIM[0]) * 100 / (LONLIM[1]-LONLIM[0]))
                if longrid < 0:
                    longrid = 0
                if longrid >= 100:
                    longrid = 100
                row_val = [idx, idy, lat, lon]
                csv_writer_trj[idx % n_trj_file].writerow(row_val)
                lat_bin = get_lat_bin(lat, nbins=10)
                lon_bin = get_lon_bin(lon, nbins=10)
                bin_set.add((lat_bin, lon_bin))
                tot_dist += dis
            for latb, lonb in sorted(bin_set):
                output_file_bin[latb][lonb].write('%s\n' % idx)
                csv_writer_nib[idx % n_trj_file].writerow([idx, latb, lonb])

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
                if latlim[0] < LATLIM[0] or latlim[1] > LATLIM[1]:
                    print('latlim', latlim)
                if lonlim[0] < LONLIM[0] or lonlim[1] > LONLIM[1]:
                    print('lonlim', lonlim)
#            if idx > 1000:
#                exit(0)

        print('latlim', latlim)
        print('lonlim', lonlim)
    output_file_idx.close()
    for outf in output_file_trj + output_file_nib:
        outf.close()
    for i in output_file_bin:
        for j in i:
            j.close()
    return

def describe_trajectory_file():
    nrows = 0
    lat_stat = [0, 0]
    lon_stat = [0, 0]
    with gzip.open('train_trj.csv.gz', 'rb') as infile:
        csv_reader = csv.reader(infile)
        labels = next(csv_reader)
        for idx, row in enumerate(csv_reader):
            if idx % 1000000 == 0:
                print('processed %d' % idx)
            row_dict = dict(zip(labels, row))
            lat = float(row_dict['LAT'])
            lon = float(row_dict['LON'])
            lat_stat[0] += lat
            lat_stat[1] += lat**2
            lon_stat[0] += lon
            lon_stat[1] += lon**2
            nrows += 1
    lat_stat[0] /= nrows
    lon_stat[0] /= nrows
    lat_stat[1] = sqrt(lat_stat[1]/nrows-lat_stat[0]**2)
    lon_stat[1] = sqrt(lon_stat[1]/nrows-lon_stat[0]**2)
    print(lat_stat, lon_stat)
    return

def describe_bins():
    bin_files = []
    for idx in range(11):
        _tmp = []
        for jdx in range(11):
            _tmp.append(gzip.open('train/train_bin_%02d_%02d.csv.gz' % (idx,
                                                                        jdx),
                                                                        'wb'))
        bin_files.append(_tmp)
    for idx in range(100):
        print('idx', idx)
        bin_dict = defaultdict(set)
        with gzip.open('train/train_trj_%02d.csv.gz' % idx, 'rb') as infile:
            csv_reader = csv.reader(infile)
            labels = next(csv_reader)
            for row in csv_reader:
                row_dict = dict(zip(labels, row))
                tdx = int(row_dict['TRAJECTORY_IDX'])
                lat = float(row_dict['LAT'])
                lon = float(row_dict['LON'])
                lat_bin = get_lat_bin(lat, nbins=10)
                lon_bin = get_lon_bin(lon, nbins=10)
                bin_dict[tdx].add((lat_bin, lon_bin))
        for tdx, val in bin_dict.items():
            for latbin, lonbin in val:
                bin_files[latbin][lonbin].write('%d\n' % tdx)

def get_trajectory(trj_idx=None, lat_bin=None, lon_bin=None,
                   fname='train_trj.csv.gz'):
    trajectory = []
    with gzip.open('train_trj.csv.gz', 'rb') as infile:
        csv_reader = csv.reader(infile)
        labels = next(csv_reader)
        for idx, row in enumerate(csv_reader):
            if idx % 1000000 == 0:
                print('processed %d' % idx)
            row_dict = dict(zip(labels, row))
            if trj_idx and int(row_dict['TRAJECTORY_IDX']) == trj_idx:
                lat = float(row_dict['LAT'])
                lon = float(row_dict['LON'])
                trajectory.append((lat, lon))
            if lat_bin and lon_bin:
                latbin = get_lat_bin(float(row_dict['LAT']))
                lonbin = get_lon_bin(float(row_dict['LON']))
                if latbin == lat_bin and lonbin == lon_bin:
                    trajectory.append(row_dict['TRAJECTORY_IDX'])
    return trajectory

if __name__ == '__main__':
    feature_extraction(is_test=False)
    feature_extraction(is_test=True)
#    describe_trajectory_file()
#    describe_bins()
