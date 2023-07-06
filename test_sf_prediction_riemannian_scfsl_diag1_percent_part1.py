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

# function for saving the scores, predictions, parameters to disk with subject ID labels
def save_scores_preds_params(
        scores_test: list, scores_train: list, preds: dict,
        params: dict, atlas: str, recon: list, edge_weight: str, batch_n: int or str,
        scrambled: bool = False):
    # set up general file labels
    if scrambled:
        fl = '_scrambled_percent_' + atlas + '_' + recon + '_' + edge_weight + '_diag1_n' + str(batch_n)
    elif not scrambled:
        fl = '_percent_' + atlas + '_' + recon + '_' + edge_weight + '_diag1_n' + str(batch_n)
    # save scores to disk
    scores_test.to_csv('/datain/dataset/scores_test' + fl + '.csv')
    scores_train.to_csv('/datain/dataset/scores_train' + fl + '.csv')
    # save preds to disk
    with open('/datain/dataset/preds' + fl + '.pkl', 'wb') as fp:
        pickle.dump(preds, fp)
    # save params to disk as pickle
    with open('/datain/dataset/params' + fl + '.pkl', 'wb') as fp:
        pickle.dump(params, fp)


for atlas in atlases:
    num_rois = 0
    if atlas == 'aal116':
        num_rois = 116

    # we need to import the fc matrices in a similar method as above!!! ^
    for edge_weight in 'dti_meanlength', 'dti_count', 'dti_volumeweighted':
        for iii in np.arange(12):
            batch_n = iii
            recon = 'dti'
            f = h5py.File('/datain/dataset/train_percent_' + recon + '_' +
                          edge_weight + '_diag1_' + str(batch_n) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            file_names_train = list(np.array(f.get('file_name_train')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_' + recon + '_' +
                          edge_weight + '_diag1_' + str(batch_n) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            file_names_test = list(np.array(f.get('file_name_test')))
            f.close()
            scores_train, scores_test, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test, save_all_scores=True)
            # add file names to scores and preds
            scores_test['file_name_test'] = file_names_test
            scores_train['file_name_train'] = file_names_train
            print(scores_test)
            print(params)
            # save outputs to disk
            save_scores_preds_params(
                scores_test, scores_train, preds, params, atlas, recon, edge_weight, batch_n, scrambled=False)
            # repeat for scrambled data
            f = h5py.File('/datain/dataset/train_scrambled_percent_' + recon + '_' +
                          edge_weight + '_diag1_' + str(batch_n) + '.h5py', 'r')
            sc_train = list(np.array(f.get('inputs')))
            fc_train = list(np.array(f.get('labels')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_' + recon + '_' +
                          edge_weight + '_diag1_' + str(batch_n) + '.h5py', 'r')
            sc_test = list(np.array(f.get('inputs')))
            fc_test = list(np.array(f.get('labels')))
            f.close()
            scores_train, scores_test, params, preds = run_sf_prediction(
                sc_train, fc_train, sc_test, fc_test, save_all_scores=True)
            # add file names to scores and preds
            scores_test['file_name_test'] = file_names_test
            scores_train['file_name_train'] = file_names_train
            print(scores_test)
            print(params)
            # save outputs to disk
            save_scores_preds_params(
                scores_test, scores_train, preds, params, atlas, recon, edge_weight, batch_n, scrambled=True)