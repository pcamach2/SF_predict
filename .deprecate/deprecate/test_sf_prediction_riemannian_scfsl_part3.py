import pickle
import csv
import h5py
import os
import sys
import numpy as np
# importing the sf_prediction package
sys.path.insert(0, '/opt/micaopen/sf_prediction')
from utils import run_sf_prediction


# ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7','schaefer400x17','schaefer400x7'
atlases = ['aal116']
# ,'power264','gordon333','aicha384','brainnetome246'

for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        num_rois = 116
    for edge_weight in 'dti_meanlength', 'dti_count', 'dti_volumeweighted':
        for iii in np.arange(25):
            # Read
            ii = iii + 50
            f = h5py.File('/datain/dataset/train_dti_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_dti_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_' + atlas +
                          '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_' + atlas +
                           '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_dti_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_dti_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_scrambled_' +
                          atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_scrambled_' + atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_scrambled_' +
                           atlas + '_dti_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
