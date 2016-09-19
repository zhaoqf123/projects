#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from math import radians, cos, sin, asin, sqrt
import copy

import pandas as pd


DIS_THRESHOLD = 15
LIST_COL_NAME = []

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Taken from http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    # convert decimal degrees to radians
    #print [lon1, lat1, lon2, lat2]
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

def sort_list(list_of_tuple):
    length_list = len(list_of_tuple)
    if length_list>=1:
        list_tuple = sorted(list_of_tuple, key=lambda x: x[1])
        return [ele for tup in list_tuple for ele in tup]
    else:
        return list_of_tuple#Acutally empty list

def cal_dis(arr_data, arr_target, dis_threshold=DIS_THRESHOLD):
    """
    Input::
        arr_data: 2-D array, 2 cols; 1st col: Lat, 2nd col: Lng
        arr_target: 2-D array, 3 cols; 1st col: id, 2nd col: Lat, 3rd col: Lng
    Output::
        return: 2-D list of list, with one column, no of rows equals arr_data
    """
    no_row = arr_data.shape[0]
    list_matched = [None]*no_row
    for i, arr_data_each_row in enumerate(arr_data):
        lat1 = arr_data_each_row[0]
        lon1 = arr_data_each_row[1]
        list_matched[i] = []
        for arr_target_each_row in arr_target:
            lat2 = arr_target_each_row[1]
            lon2 = arr_target_each_row[2]
            dis = haversine(lon1, lat1, lon2, lat2)
            if dis <= dis_threshold:
                city_id = arr_target_each_row[0]
                list_matched[i].append((city_id, round(dis, 2)))
        list_matched[i] = sort_list(list_matched[i])
    return list_matched


def cal_dis_2(arr_data, arr_target, dis_threshold=DIS_THRESHOLD):
    """
    Input::
        arr_data: 2-D array, 2 cols; 1st col: Lat, 2nd col: Lng
        arr_target: 2-D array, 3 cols; 1st col: id, 2nd col: Lat, 3rd col: Lng
    Output::
        return: 2-D list of list, with one column, no of rows equals arr_data; each element list is of ID and dis
                2-D list of list, with one column, no of rows equals arr_data; each element list is of ID ONLY
    """
    list_matched = cal_dis(arr_data, arr_target, dis_threshold)
    list_matched_id_only = [[ele[i] for i in xrange(0, len(ele), 2)] for ele in list_matched]
    return list_matched, list_matched_id_only


def cal_dis_closest_city(arr_data, arr_target):
    """
    Input::
        arr_data: 2-D array, 2 cols; 1st col: Lat, 2nd col: Lng
        arr_target: 2-D array, 4 cols; 1st col: State, 2nd col: City, 3rd col: Lat, 4th col: Lng
    Output::
        return: 2-D list of list, with 5 cols, 4 of them like arr_target, one is distance; no of rows equals arr_data
    """
    no_row = arr_data.shape[0]
    list_matched = [None]*no_row
    for i, arr_data_each_row in enumerate(arr_data):
        lat1 = arr_data_each_row[0]
        lon1 = arr_data_each_row[1]
        dis = 1000
        list_matched[i] = [dis, "X", "X", 0.0, 0.0]
        for arr_target_each_row in arr_target:
            lat2 = arr_target_each_row[2]
            lon2 = arr_target_each_row[3]
            dis_tmp = haversine(lon1, lat1, lon2, lat2)
            if dis_tmp < dis:
                dis = dis_tmp
                tmp_list = [round(dis,2)] + list(arr_target_each_row)
                list_matched[i] = tmp_list
    return list_matched


class CityMap(object):
    def __init__(self, df_data, df_ref):
        """
        Map cities from two different sources based on the lat/lon
        Input::
            df_data: data frame, 2 columns, lat and lon
            df_ref: data frame, 2 column, city_id, lat and lon
        """
        self._df_data = df_data
        self._df_ref = df_ref
        self._arr_data = df_data.values
        self._arr_ref = df_ref.values

    
    def _create_col(self, no_col):
        global LIST_COL_NAME
        list_col_name = [None]*no_col
        for i in xrange(no_col/2):
            list_col_name[2*i], list_col_name[2*i+1] = "City_" + str(i+1), "Dis_" + str(i+1)
        if len(LIST_COL_NAME) < no_col:
            LIST_COL_NAME = copy.deepcopy(list_col_name)
        return list_col_name

    def _create_df_from_list(self, list_city_dis_matched):
        df = pd.DataFrame(list_city_dis_matched)
        no_col = df.shape[1]
        col_name = self._create_col(no_col)
        df.columns = col_name
        return df

    def get_match_result(self):
        list_city_dis_matched = cal_dis(self._arr_data, self._arr_ref)
        df_city_matched = self._create_df_from_list(list_city_dis_matched)

        df_city_matched.index = self._df_data.index
        df_new = pd.concat([self._df_data.copy(), df_city_matched], axis=1)
        if len(self._df_data) != len(df_new):
            print(self._df_data.shape,"\n", df_new.shape)
            raise ValueError("Not of the same length")
        return df_new




