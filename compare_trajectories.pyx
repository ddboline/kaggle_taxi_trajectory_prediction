# -*- coding: utf-8 -*-
"""
Created on Tue May 12 07:45:58 2015

@author: ddboline
"""
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport asin, sin, cos, fabs, sqrt

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

cdef double pi = np.pi

def lat_lon_box(double lat, double lon, double dist):
    cdef double adlat=0, adlon=0
    lat_lon_box_(lat, lon, dist, adlat, adlon)
    return adlat, adlon

cdef void lat_lon_box_(double lat, double lon, double dist, double adlat, double adlon):
    cdef double r_rearch, d_2r, dlat=0, dlon=0
    with cython.nogil:
        r_earth = 6371.
        d_2r = dist/(2.*r_earth)
        dlat = 2. * (d_2r)
        dlon = 2. * asin((sin(d_2r))/(cos(lat)))
        dlat *= 180./pi
        dlon *= 180./pi
        adlat = fabs(dlat)
        adlon = fabs(dlon)

def haversine_distance(double lat1, double lon1, double lat2, double lon2):
    return haversine_distance_(lat1, lon1, lat2, lon2)

cdef double haversine_distance_(double lat1, double lon1, double lat2, double lon2):
    cdef double r_earch, dlat, dlon, dist
    with cython.nogil:
        r_earth = 6371.
        dlat = fabs(lat1-lat2)*pi/180.
        dlon = fabs(lon1-lon2)*pi/180.
        lat1 *= pi/180.
        lat2 *= pi/180.
        dist = 2. * r_earth * asin(sqrt(sin(dlat/2.)**2 +
                                        cos(lat1)*cos(lat2)*sin(dlon/2.)**2))
    return dist

@cython.boundscheck(False)
def compare_trajectories(np.ndarray[DTYPE_t, ndim=2] test_trj, np.ndarray[DTYPE_t, ndim=2] train_trj):
    cdef int n_common = 0, n_common_tr = 0
    cdef double test_lat, test_lon, train_lat, train_lon, dis, dlat=0, dlon=0
    cdef unsigned int idx, jdx
    with cython.nogil:
        for idx in range(test_trj.shape[0]):
            test_lat = test_trj[idx, 0]
            test_lon = test_trj[idx, 1]
            lat_lon_box_(test_lat, test_lon, 0.1, dlat, dlon)
            n_common_tr = 0
            for jdx in range(train_trj.shape[0]):
                train_lat = train_trj[jdx, 0]
                train_lon = train_trj[jdx, 1]
                if fabs(train_lat-test_lat) > dlat or \
                        fabs(train_lon-test_lon) > dlon:
                    continue
                dis = haversine_distance(test_lat, test_lon, train_lat, train_lon)
                if dis < 0.05:
                    n_common_tr += 1
            if n_common_tr > 0:
                n_common += 1
    return n_common
