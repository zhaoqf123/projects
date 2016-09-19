# encoding: utf-8
from __future__ import print_function

from datetime import datetime

from fuzzywuzzy import utils as utils_clean
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
reload(fuzz)

import pandas as pd

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer

import utils_coding
reload(utils_coding)


"""
This utils is for dealing with job title matters, for example, matching raw job titles to
GOLDEN job titles
"""

#ST = LancasterStemmer()
ST = PorterStemmer()

RATE_COL_I = 0.2
RATE_COL_II = 1.0 - RATE_COL_I

def agg_2_score(score_I, score_II):
    return score_I*RATE_COL_I + score_II*RATE_COL_II

FUNC_MATCH = fuzz.UWRatio
FUNC_AGG = agg_2_score

def stem_jobtitle(jobtitle):
    jobtitle = utils_clean.full_process(jobtitle)
    list_token = jobtitle.split()
    list_token = [ST.stem(ele) for ele in list_token]
    return u" ".join(list_token)


class JobTitleMatch(object):
    """Match raw job titles into golden job titles"""
    def __init__(self, df_data, df_ref):
        """
        Input::
            df_data: data frame, 2 columns, <jobtitle_translated>, <jobcategory>
            df_ref:  data frame, 2 columns, <JobTitle>, <JobCategory>
        """
        # col_I = ["jobcategory", "jobtitle_translated"]
        col_II = ["JobCategory", "JobTitle"]
        # self._df_data = df_data[col_I].copy()
        # self._df_data.columns = col_II
        self._df_data = df_data[col_II].copy()
        self._df_ref = df_ref[col_II].copy()


    def _compare_series(self, ds, ds_ref):
        score = [0,0]
        score[0] = FUNC_MATCH(ds[0], ds_ref[0])
        score[1] = FUNC_MATCH(ds[1], ds_ref[1])
        return FUNC_AGG(*score)

    def _compare_col(self, ds, df_ref):
        """
        ds: pd series if param raw=False; if True, it is <np array>
        df_ref: golden reference for raw data to be mapped to
        """
        df_ref["Score"] = df_ref.apply(lambda ds_ref: self._compare_series(ds, ds_ref), axis=1, raw=True)
        idx = df_ref["Score"].idxmax(axis=0)
        return df_ref.ix[idx]

    def compare_two_col(self, title_stem=True):
        """Comapre two columns to get a match"""
        df_data = self._df_data.copy()
        df_ref = self._df_ref.copy()
        col = ["JobCategory", "JobTitle", "Score"]
        df_data["Score"] = [0]*len(df_data)
        if title_stem:
            print("Start Stem Job Titles!")
            df_data["JobTitleStemmed"] = df_data["JobTitle"].apply(stem_jobtitle)
            df_ref["JobTitleStemmed"] = df_ref["JobTitle"].apply(stem_jobtitle)
            col = ["JobCategory", "JobTitleStemmed", "Score"]
            self._df_ref = df_ref.copy()
            df_ref = df_ref[["JobCategory", "JobTitleStemmed"]]
        start_time = datetime.now()
        df_data[["JobCategoryGolden", "JobTitleGolden", "Score"]] = df_data[col].apply(lambda ds: self._compare_col(ds, df_ref), axis=1, raw=True)
        end_time = datetime.now()
        print("Time consumed for matching is {}".format((end_time-start_time).seconds))
        return self._merge_match_results(df_data)

    def _merge_match_results(self, df_data):
        df_merge = pd.merge(df_data, self._df_ref, how="left", left_on=["JobCategoryGolden", "JobTitleGolden"],
            right_on=["JobCategory", "JobTitleStemmed"], suffixes=("", "_GOLDEN"))
        col = [u'JobCategory', u'JobTitle', 'JobTitleStemmed', 'Score', u'JobCategory_GOLDEN', u'JobTitle_GOLDEN', 'JobTitleStemmed_GOLDEN']
        return df_merge[col]






































