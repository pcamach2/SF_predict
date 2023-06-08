from sklearn import svm
# from nilearn.decoding import Decoder
import matplotlib.pyplot as plt
# from nilearn.plotting import show
import pandas as pd
import os,sys
import numpy as np
import scipy.io as sio
import pymanopt
import brainspace
import sklearn
# importing the sf_prediction package
sys.path.insert(0, '/opt/micaopen/sf_prediction')
from utils import run_sf_prediction
from sklearn.model_selection import train_test_split
import h5py
import csv
from scipy.sparse import csr_matrix
import pickle

os.chdir('/datain')
#struct conn matrices
matfilespath = '/datain/matfiles_dti_volwei_06052023_/'  # '/datain/matfiles_dti_count_11182022_/'
inpath_files = os.listdir(matfilespath)
part_num = len(inpath_files)

# RSFC matrices

fcmatfilespath = '/datain/matfiles_aroma_06052023/'
inpath_files_fc = os.listdir(fcmatfilespath)
part_num_fc = len(inpath_files_fc)

#behavioral = pd.read_csv('/datain/group_data.csv')
# y = behavioral['fft_stair_ds_tester1'].to_numpy()
# y = behavioral['fft_4step_t1'].to_numpy()
#y = behavioral['peakvo2_ml_gxt'].to_numpy()

scripts = '/datain/atlas_ids/'
# update with new atlases added as needed
# ^ custom atlases can be used in QSIPrep as part of a custom reconstruction workflow
atlases = ['aal116'] # ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7','schaefer400x17','schaefer400x7','aal116','power264','gordon333','aicha384','brainnetome246'
# mean_strength_mat = np.zeros((part_num,len(atlases)))
# global_efficiency_mat = np.zeros((part_num,len(atlases))) # # atlas_scores = {}
# atlas_chance_scores = {}
# 
print('all atlases included for analyses, please correct for multiple comparisons accordingly')
print('populating atlas region names dataframes')
# aal116_keys = pd.read_csv(scripts+'aal116_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# power264_keys = pd.read_csv(scripts+'power264_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# gordon333_keys = pd.read_csv(scripts+'gordon333_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# aicha384_keys = pd.read_csv(scripts+'aicha384_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# brainnetome246_keys = pd.read_csv(scripts+'brainnetome246_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer100x17_keys = pd.read_csv(scripts+'schaefer100x17_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer100x7_keys = pd.read_csv(scripts+'schaefer100x7_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer200x17_keys = pd.read_csv(scripts+'schaefer200x17_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer200x7_keys = pd.read_csv(scripts+'schaefer200x7_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer400x17_keys = pd.read_csv(scripts+'schaefer400x17_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])
# schaefer400x7_keys = pd.read_csv(scripts+'schaefer400x7_origLUT.txt', delimiter='\t', header = None, names = ['roi_num', 'roi_id'])

for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        # labels = aal116_keys['roi_id']
        # keys = aal116_keys
        num_rois = 116

    # we need to import the fc matrices in a similar method as above!!! ^
#    for edge_weight in 'count_pass','count_end','ncount_pass','ncount_end','mean_length_pass','mean_length_end':
    for edge_weight in 'dti_meanlength', 'dti_count','dti_volumeweighted': # ,"dti_meanlength":
    # if num_rois == 116: # hot fix
        # edge_weight = 'dti_volumeweighted' # "dti_count"
        for iii in np.arange(25):
            # Read
            ii = iii
            f = h5py.File('/datain/dataset/train_percent_dti_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_dti_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key,val])
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_percent_dti_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_dti_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_scrambled_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_scrambled_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_scrambled_percent_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key,val])



