import pickle
import csv
import h5py
import sys
import numpy as np
# importing the sf_prediction package
sys.path.insert(0, '/opt/micaopen/sf_prediction')
from utils import run_sf_prediction

scripts = '/datain/atlas_ids/'
atlases = ['aal116']  # ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7','schaefer400x17','schaefer400x7','aal116','power264','gordon333','aicha384','brainnetome246'


for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        num_rois = 116

    # we need to import the fc matrices in a similar method as above!!! ^
    for edge_weight in 'count_sum', 'volume_weighted_count_sum', 'mean_length_sum':
        for iii in np.arange(25):
            ii = iii
            f = h5py.File('/datain/dataset/train_percent_gqi_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_gqi_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_percent_' + atlas +
                          '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_percent_' + atlas +
                           '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_percent_gqi_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_gqi_' +
                          edge_weight + '_diag1_' + str(ii) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test)
            print(scores)
            print(params)
            scores.to_csv('/datain/dataset/scores_scrambled_percent_' +
                          atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv')
            # save preds to disk
            with open('/datain/dataset/preds_percent_' + atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.pkl', 'wb') as fp:
                pickle.dump(preds, fp)
            f = csv.writer(open('/datain/dataset/params_scrambled_percent_' +
                           atlas + '_gqi_' + edge_weight + '_diag1_n' + str(ii) + '.csv', "w"))
            for key, val in params.items():
                f.writerow([key, val])
