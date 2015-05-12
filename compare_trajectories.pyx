# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:45:58 2015

@author: ddboline
"""
import numpy as np
cimport numpy as np

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

def lat_lon_box(double lat, double lon, double dist):
    cdef double r_earth = 6371.
    cdef double d_2r = dist/(2.*r_earth)
    cdef double dlat = 2. * (d_2r)
    cdef double dlon = 2. * np.arcsin((np.sin(d_2r))/(np.cos(lat)))
    dlat *= 180./np.pi
    dlon *= 180./np.pi
    return abs(dlat), abs(dlon)

def haversine_distance(double lat1, double lon1, double lat2, double lon2):
    cdef double r_earth = 6371.
    cdef double dlat = abs(lat1-lat2)*np.pi/180.
    cdef double dlon = abs(lon1-lon2)*np.pi/180.
    lat1 *= np.pi/180.
    lat2 *= np.pi/180.
    cdef double dist = 2. * r_earth * np.arcsin(np.sqrt(np.sin(dlat/2.)**2 +
                                      np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.)**2))
    return dist

def compare_trajectories(np.ndarray test_trj, np.ndarray train_trj):
    n_common = 0
    for idx in range(test_trj.shape[0]):
        test_lat = test_trj[idx, 0]
        test_lon = test_trj[idx, 1]
        dlat, dlon = lat_lon_box(test_lat, test_lon, 0.2)
        n_common_tr = 0
        for jdx in range(train_trj.shape[0]):
            train_lat = train_trj[jdx, 0]
            train_lon = train_trj[jdx, 1]
            if abs(train_lat-test_lat) > dlat or \
                    abs(train_lon-test_lon) > dlon:
                continue
            dis = haversine_distance(test_lat, test_lon, train_lat, train_lon)
            if dis < 0.1:
                n_common_tr += 1
        if n_common_tr > 0:
            n_common += 1
    return n_common
