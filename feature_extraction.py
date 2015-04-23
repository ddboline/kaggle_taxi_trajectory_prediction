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
    latlons = []
    for latlon in polyline_str.replace('[[','').replace(']]','').split('],['):
        lat, lon = map(float, latlon.split(','))
        latlons.append((lat, lon))
    return latlons
    
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
    
    if is_test:
        output_file = gzip.open('test_fe.csv.gz', 'wb')
    else:
        output_file = gzip.open('train_fe.csv.gz', 'wb')
    
    csv_writer = csv.writer(output_file)
    
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
        csv_writer.writerow(new_labels)
        for idx, row in enumerate(csv_reader):
            row_dict = dict(zip(labels, row))
            latlon_points = split_polyline(row_dict['POLYLINE'])
            if len(latlon_points) == 0:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = ('nan', 'nan')
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = ('nan', 'nan')
            else:
                row_dict['ORIGIN_LAT'], row_dict['ORIGIN_LON'] = latlon_points[0]
                row_dict['DEST_LAT'], row_dict['DEST_LON'] = latlon_points[-1]
            tot_dist = total_distance(latlon_points)
            row_dict['NUMBER_POINTS'] = len(latlon_points)
            row_dict['TOTAL_DISTANCE'] = tot_dist

            row_val = [row_dict[col] for col in new_labels]
            csv_writer.writerow(row_val)
            print row_val
            raw_input()
            if idx % 10000 == 0:
                print 'processed %d' % idx
    output_file.close()
    return

if __name__ == '__main__':
    feature_extraction(is_test=False)
    feature_extraction(is_test=True)
