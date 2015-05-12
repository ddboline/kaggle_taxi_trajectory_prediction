# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:45:58 2015

@author: ddboline
"""
from feature_extraction import lat_lon_box, haversine_distance

def compare_trajectories(test_trj, train_trj):
    n_common = 0
    for test_lat, test_lon in test_trj:
        dlat, dlon = lat_lon_box(test_lat, test_lon, 0.2)
        n_common_tr = 0
        for train_lat, train_lon in train_trj:
            if abs(train_lat-test_lat) > dlat or \
                    abs(train_lon-test_lon) > dlon:
                continue
            dis = haversine_distance(test_lat, test_lon, train_lat, train_lon)
            if dis < 0.1:
                n_common_tr += 1
        if n_common_tr > 0:
            n_common += 1
    return n_common
