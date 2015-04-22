#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 15:36:00 2015

@author: ddboline
"""
import gzip
import csv
import pandas as pd

from load_data import haversine_distance

def split_polyline(polyline_str):
    poly_struct = eval(polyline_str.replace('[','(').replace(']',')'))
    return poly_struct

def total_distance(latlon_points):
    if len(latlon_points) <= 2:
        return 0.
    lat0, lon0 = latlon_points[0]
    tot_dist = 0
    for lat, lon in latlon_points[1:]:
        tot_dist += haversine_distance(lat0, lon0, lat, lon)
        lat0, lon0 = lat, lon
    return tot_dist

def feature_extraction(is_test=False):
    metadata_df = pd.read_csv('metaData_taxistandsID_name_GPSlocation.csv.gz',
                              compression='gzip')
                              
    print metadata_df.columns
    
    output_files = []
    if is_test:
        output_files = {ct: gzip.open('test_fe_%s.csv.gz' % ct, 'wb') 
                        for ct in ('A', 'B', 'C')}
    else:
        output_files = {ct: gzip.open('train_fe_%s.csv.gz' % ct, 'wb') 
                        for ct in ('A', 'B', 'C')}
    
    csv_writers = {ct: csv.writer(f) for (ct, f) in output_files.items()}
    
    input_file = 'train.csv.gz'
    if is_test:
        input_file = 'test.csv.gz'
    
    with gzip.open(input_file, 'rb') as infile:
        csv_reader = csv.reader(infile)
        labels = next(csv_reader)
        new_labels = ['TRIP_ID', 'CALL_TYPE', 'ORIGIN_CALL', 'ORIGIN_STAND', 
                      'TAXI_ID', 'TIMESTAMP', 'DAY_TYPE', 'MISSING_DATA', 
                      'ORIGIN_LAT', 'ORIGIN_LON',
                      'NUMBER_POINTS',
                      'TOTAL_DISTANCE',
                      'DEST_LAT', 'DEST_LON']
        for cwr in csv_writers.values():
            cwr.writerow(new_labels)
        for idx, row in enumerate(csv_reader):
            row_dict = dict(zip(labels, row))
            latlon_points = split_polyline(row_dict['POLYLINE'])
            if len(latlon_points) == 0:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = ('nan', 'nan')
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = ('nan', 'nan')
            elif len(latlon_points) > 2:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = latlon_points[0]
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = latlon_points[-1]
            else:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = latlon_points
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = latlon_points
            tot_dist = total_distance(latlon_points)
            row_dict['NUMBER_POINTS'] = len(latlon_points)
            row_dict['TOTAL_DISTANCE'] = tot_dist

            row_val = [row_dict[col] for col in new_labels]
            csv_writers[row_dict['CALL_TYPE']].writerow(row_val)
            if idx % 10000 == 0:
                print 'processed %d' % idx
    for inf in output_files.values():
        inf.close()
    return

if __name__ == '__main__':
    feature_extraction(is_test=False)
    feature_extraction(is_test=True)
