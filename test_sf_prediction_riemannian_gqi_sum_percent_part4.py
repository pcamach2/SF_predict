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

# #struct conn matrices
# matfilespath = '/datain/matfiles_gqi_11182022_/'
# inpath_files = os.listdir(matfilespath)
# part_num = len(inpath_files)

# # RSFC matrices

# fcmatfilespath = '/datain/matfiles_aroma_11182022/'
# inpath_files_fc = os.listdir(fcmatfilespath)
# part_num_fc = len(inpath_files_fc)

# behavioral = pd.read_csv('/datain/group_data.csv')
# # y = behavioral['fft_stair_ds_tester1'].to_numpy()
# # y = behavioral['fft_4step_t1'].to_numpy()
# y = behavioral['peakvo2_ml_gxt'].to_numpy()

scripts = '/datain/atlas_ids/'
# update with new atlases added as needed
# ^ custom atlases can be used in QSIPrep as part of a custom reconstruction workflow
atlases = ['aal116'] # ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7','schaefer400x17','schaefer400x7','aal116','power264','gordon333','aicha384','brainnetome246'


for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        num_rois = 116


    # we need to import the fc matrices in a similar method as above!!! ^
    for edge_weight in 'count_sum','ncount_sum','mean_length_sum','gfa_sum':
        for iii in np.arange(25):
            ii = iii + 75
            f = h5py.File('/datain/dataset/train_percent_gqi_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_gqi_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key,val])
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_percent_gqi_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_gqi_' + edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_scrambled_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_scrambled_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key,val])



