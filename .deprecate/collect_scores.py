import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys
import numpy as np
import scipy.io as sio
import h5py
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def collect_scores(data1, data2, data3, column_id, weight_type):
    """ Saves a csv of scores from three pandas DataFrames matching a column_id using seaborn library. 
    Args: 
    data1 (pandas.DataFrame): The input DataFrame 
    data2 (pandas.DataFrame): The input DataFrame 
    data3 (pandas.DataFrame): The input DataFrame 
    column_id (int): The column value for which the line plot will be created (e.g. dataset == 001)
    weight_type (str): The type of weighting used for the scores (e.g. count, volume_weighted, mean_path_length)
    Returns: None 
    """
    column_id = int(column_id)
    data1.rename(columns={'Pearson_Walk_1': '1', 'Pearson_Walk_2': '2', 'Pearson_Walk_3': '3', 'Pearson_Walk_4': '4', 'Pearson_Walk_5': '5', 'Pearson_Walk_6': '6',
                 'Pearson_Walk_7': '7', 'Pearson_Walk_8': '8', 'Pearson_Walk_9': '9', 'Pearson_Walk_10': '10', 'Batch_Number': 'Batch_Number'}, inplace=True)
    data2.rename(columns={'Pearson_Walk_1': '1', 'Pearson_Walk_2': '2', 'Pearson_Walk_3': '3', 'Pearson_Walk_4': '4', 'Pearson_Walk_5': '5', 'Pearson_Walk_6': '6',
                 'Pearson_Walk_7': '7', 'Pearson_Walk_8': '8', 'Pearson_Walk_9': '9', 'Pearson_Walk_10': '10', 'Batch_Number': 'Batch_Number'}, inplace=True)
    data3.rename(columns={'Pearson_Walk_1': '1', 'Pearson_Walk_2': '2', 'Pearson_Walk_3': '3', 'Pearson_Walk_4': '4', 'Pearson_Walk_5': '5', 'Pearson_Walk_6': '6',
                 'Pearson_Walk_7': '7', 'Pearson_Walk_8': '8', 'Pearson_Walk_9': '9', 'Pearson_Walk_10': '10', 'Batch_Number': 'Batch_Number'}, inplace=True)
    data1 = data1[data1['Batch_Number'] == column_id]
    data2 = data2[data2['Batch_Number'] == column_id]
    data3 = data3[data3['Batch_Number'] == column_id]

    data1m = data1.drop(
        columns=['Participant_Number', 'Batch_Number'], inplace=False)
    data2m = data2.drop(
        columns=['Participant_Number', 'Batch_Number'], inplace=False)
    data3m = data3.drop(
        columns=['Participant_Number', 'Batch_Number'], inplace=False)
    if weight_type == 'count':
        msmt_name = 'MSMT CSD SIFT2 Streamline Count'
        gqi_name = 'GQI Streamline Count'
        dti_name = 'DTI Streamline Count'
    elif weight_type == 'volume_weighted':
        msmt_name = 'MSMT CSD SIFT2 Node Volume Weighted Streamline Count'
        gqi_name = 'GQI Node Volume Weighted Streamline Count'
        dti_name = 'DTI Node Volume Weighted Streamline Count'
    elif weight_type == 'mean_path_length':
        msmt_name = 'MSMT CSD Mean Length'
        gqi_name = 'GQI Mean Length'
        dti_name = 'DTI Mean Length'
    # note here that we will be using the percent of streamlines that each edge represents from the total number of streamlines connecting that node
    data1m['Recon'] = msmt_name
    data2m['Recon'] = gqi_name
    data3m['Recon'] = dti_name
    df = pd.concat([data1m, data2m, data3m], axis=0, ignore_index=True)

    df.to_csv(
        f"/datain/dataset/{weight_type}_all_percent_batch{column_id}.csv")


# set path for data - this is /datain/dataset for mounting to docker/singularity container
main_path = '/datain/dataset'


# initialize data structures
dti_mean_path_length_all = pd.DataFrame()
dti_volume_weighted_count_all = pd.DataFrame()
dti_count_all = pd.DataFrame()

msmt_sift_radius2_count_all = pd.DataFrame()
msmt_sift_radius2_invnodevol_count_all = pd.DataFrame()
msmt_radius2_count_all = pd.DataFrame()
msmt_radius2_meanlength_all = pd.DataFrame()

gqi_mean_length_sum_all = pd.DataFrame()
gqi_ncount_sum_all = pd.DataFrame()
gqi_count_sum_all = pd.DataFrame()
gqi_gfa_sum_all = pd.DataFrame()
gqi_mean_length_pass_all = pd.DataFrame()
gqi_ncount_pass_all = pd.DataFrame()
gqi_count_pass_all = pd.DataFrame()
gqi_gfa_pass_all = pd.DataFrame()
gqi_mean_length_end_all = pd.DataFrame()
gqi_ncount_end_all = pd.DataFrame()
gqi_count_end_all = pd.DataFrame()
gqi_gfa_end_all = pd.DataFrame()


# import clean csvs
# DTI

files_mean_path_length = Path(main_path).glob(
    'scores*percent*dti*mean*diag1*clean*csv')
files_volume_weighted_count = Path(main_path).glob(
    'scores*percent_aal116_dti_dti_volumeweighted_diag1_???_clean.csv')
files_count = Path(main_path).glob(
    'scores*percent*_aal116_dti_dti_count_diag1_???_clean.csv')

dfs = list()
for f in files_volume_weighted_count:
    data = pd.read_csv(f)
    dfs.append(data)
dti_volume_weighted_count_all = pd.concat(dfs, ignore_index=True)

dfs = list()
for f in files_count:
    data = pd.read_csv(f)
    dfs.append(data)
dti_count_all = pd.concat(dfs, ignore_index=True)

dti_mean_path_length_all.describe()
dti_volume_weighted_count_all.describe()
dti_count_all.describe()

# MSMT

files_sift_radius2_count = Path(main_path).glob(
    'scores*percent*msmt_sift_radius2_count_diag1*clean*csv')
files_sift_radius2_invnodevol_count = Path(main_path).glob(
    'scores*percent*msmt_sift_invnodevol_radius2_count_diag1*clean*csv')
files_radius2_count = Path(main_path).glob(
    'scores*percent*msmt_radius2_count_diag1*clean*csv')
files_radius2_meanlength = Path(main_path).glob(
    'scores*percent*msmt_radius2_meanlength_diag1*clean*csv')

dfs = list()
for f in files_radius2_meanlength:
    data = pd.read_csv(f)
    dfs.append(data)
msmt_radius2_meanlength_all = pd.concat(dfs, ignore_index=True)
msmt_radius2_meanlength_all.describe()
dfs = list()
for f in files_sift_radius2_invnodevol_count:
    data = pd.read_csv(f)
    dfs.append(data)
msmt_sift_radius2_invnodevol_count_all = pd.concat(dfs, ignore_index=True)
msmt_sift_radius2_invnodevol_count_all.describe()
dfs = list()
for f in files_sift_radius2_count:
    data = pd.read_csv(f)
    dfs.append(data)
msmt_sift_radius2_count_all = pd.concat(dfs, ignore_index=True)
msmt_sift_radius2_count_all.describe()
dfs = list()
for f in files_radius2_count:
    data = pd.read_csv(f)
    dfs.append(data)
msmt_radius2_count_all = pd.concat(dfs, ignore_index=True)
msmt_radius2_count_all.describe()

# GQI

files_mean_length_sum = Path(main_path).glob(
    'scores*percent*gqi*mean_length_sum*clean*csv')
files_mean_length_pass = Path(main_path).glob(
    'scores*percent*gqi*mean_length_pass*clean*csv')
files_mean_length_end = Path(main_path).glob(
    'scores*percent*gqi*mean_length_end*clean*csv')
files_ncount_sum = Path(main_path).glob(
    'scores*percent*gqi_ncount_sum*clean*csv')
files_ncount_pass = Path(main_path).glob(
    'scores*percent*gqi_ncount_pass*clean*csv')
files_ncount_end = Path(main_path).glob(
    'scores*percent*gqi_ncount_end*clean*csv')
files_count_sum = Path(main_path).glob(
    'scores*percent*gqi_count_sum*clean*csv')
files_count_pass = Path(main_path).glob(
    'scores*percent*gqi_count_pass*clean*csv')
files_count_end = Path(main_path).glob(
    'scores*percent*gqi_count_end*clean*csv')
# files_gfa_sum = Path(main_path).glob('scores*gqi_gfa_sum*clean*csv')
# files_gfa_pass = Path(main_path).glob('scores*gqi_gfa_pass*clean*csv')
# files_gfa_end = Path(main_path).glob('scores*gqi_gfa_end*clean*csv')

dfs = list()
for f in files_mean_length_sum:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_mean_length_sum_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_mean_length_pass:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_mean_length_pass_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_mean_length_end:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_mean_length_end_all = pd.concat(dfs, ignore_index=True)

dfs = list()
for f in files_ncount_sum:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_ncount_sum_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_ncount_pass:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_ncount_pass_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_ncount_end:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_ncount_end_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_count_sum:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_count_sum_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_count_pass:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_count_pass_all = pd.concat(dfs, ignore_index=True)
dfs = list()
for f in files_count_end:
    data = pd.read_csv(f)
    dfs.append(data)
gqi_count_end_all = pd.concat(dfs, ignore_index=True)

gqi_mean_length_sum_all.T.describe()
gqi_mean_length_pass_all.T.describe()
gqi_mean_length_end_all.T.describe()
gqi_ncount_sum_all.T.describe()
gqi_ncount_pass_all.T.describe()
gqi_ncount_end_all.T.describe()
gqi_count_sum_all.T.describe()
gqi_count_pass_all.T.describe()
gqi_count_end_all.T.describe()
# gqi_gfa_sum_all.T.describe()
# gqi_gfa_pass_all.T.describe()
# gqi_gfa_end_all.T.describe()

for i in np.arange(50):
    colI = '0'
    if i < 10:
        colI = '00' + str(i)
    if i < 100:
        colI = '0' + str(i)
    collect_scores(msmt_sift_radius2_count_all, gqi_count_sum_all,
                   dti_count_all, column_id=colI, weight_type='count')
    collect_scores(msmt_sift_radius2_invnodevol_count_all,
                   gqi_ncount_sum_all, dti_volume_weighted_count_all, column_id=colI, weight_type='volume_weighted')
    collect_scores(msmt_radius2_meanlength_all, gqi_mean_length_sum_all,
                   dti_mean_path_length_all, column_id=colI, weight_type='mean_length')
