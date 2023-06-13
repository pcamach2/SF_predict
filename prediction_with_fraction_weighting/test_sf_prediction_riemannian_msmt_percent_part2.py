import pickle
import csv
import h5py
from utils import run_sf_prediction
import os
import sys
import numpy as np
# importing the sf_prediction package
sys.path.insert(0, '/opt/micaopen/sf_prediction')

scripts = '/datain/atlas_ids/'
atlases = ['aal116']

for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        num_rois = 116

    for edge_weight in 'sift_radius2_count', 'sift_invnodevol_radius2_count', 'radius2_count', 'radius2_meanlength':
        for iii in np.arange(25):
            ii = iii + 25
            f = h5py.File('/datain/dataset/train_percent_msmt_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_msmt_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_percent_' + atlas +
                          '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_percent_' + atlas +
                           '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_percent_msmt_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_msmt_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_scrambled_percent_' +
                          atlas + '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_scrambled_percent_' +
                           atlas + '_msmt_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
