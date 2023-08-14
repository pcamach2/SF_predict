# usage: test_sf_prediction_riemannian_gqi_sum_percent_save_v5.py <sc_connectomes_dir> <functional_connectomes_dir> <no_scramble>
import bct
import h5py
from sklearn.model_selection import train_test_split
import os
import numpy as np
import scipy.io as sio
import argparse

# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('sc_connectomes_dir', help='path to structural connectomes directory', type=str, required=True)
parser.add_argument('fc_connectomes_dir', help='path to functional connectomes directory', type=str, required=True)
parser.add_argument('no_scramble', help='skip production and testing of scrambled connectomes', type=bool, default=True)
args = parser.parse_args()
sc_connectomes_dir = args.sc_connectomes_dir
fc_connectomes_dir = args.fc_connectomes_dir
no_scramble = args.no_scramble

# struct conn matrices
matfilespath = '/datain/' + sc_connectomes_dir
inpath_files = sorted(os.listdir(matfilespath))
part_num = len(inpath_files)

# RSFC matrices
fcmatfilespath = '/datain/' + fc_connectomes_dir
inpath_files_fc = sorted(os.listdir(fcmatfilespath))
part_num_fc = len(inpath_files_fc)

scripts = '/datain/atlas_ids/'
atlases = ['aal116']  # ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7',
                      # 'schaefer400x17','schaefer400x7','aal116','power264','gordon333','aicha384','brainnetome246'

for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        # labels = aal116_keys['roi_id']
        # keys = aal116_keys
        num_rois = 116
    print("Concatenating all structural connectomes for %s parcellation" % atlas)
    # GQI
    allsub_mat_gfa_pass = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_gfa_end = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_volume_weighted_count_pass = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_volume_weighted_count_end = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_count_pass = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_count_end = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_mean_length_pass = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_mean_length_end = np.zeros((part_num, num_rois, num_rois))
    # CSD + SIFT2
    allsub_mat_sift_radius2_count = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_sift_invnodevol_radius2_count = np.zeros(
        (part_num, num_rois, num_rois))
    allsub_mat_radius2_count = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_radius2_meanlength = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_fcon = np.zeros((part_num, num_rois, num_rois))

    i = 0
    for matfile in inpath_files:
        mat = sio.loadmat(matfilespath + matfile)  # load mat-file
        print(matfile)
        if 'msmtconnectome' in matfile:
            # variable in mat file
            mdata_sift_radius2_count = mat[atlas +
                                           '_sift_radius2_count_connectivity']
            # variable in mat file
            mdata_sift_invnodevol_radius2_count = mat[atlas +
                                                      '_sift_invnodevol_radius2_count_connectivity']
            # variable in mat file
            mdata_radius2_count = mat[atlas+'_radius2_count_connectivity']
            # variable in mat file
            mdata_radius2_meanlength = mat[atlas +
                                           '_radius2_meanlength_connectivity']
            # set 3d matrices
            allsub_mat_sift_radius2_count[i, :, :] = mdata_sift_radius2_count
            allsub_mat_sift_invnodevol_radius2_count[i,
                                                     :, :] = mdata_sift_invnodevol_radius2_count
            allsub_mat_radius2_count[i, :, :] = mdata_radius2_count
            allsub_mat_radius2_meanlength[i, :, :] = mdata_radius2_meanlength
            i = i + 1
        elif 'gqi' in matfile:
            # variable in mat file
            mdata_gfa_pass = mat[atlas + '_gfa_pass_connectivity']
            mdata_gfa_end = mat[atlas + '_gfa_end_connectivity']
            # variable in mat file
            mdata_volume_weighted_count_pass = mat[atlas + '_volume_weighted_count_pass_connectivity']
            mdata_volume_weighted_count_end = mat[atlas + '_volume_weighted_count_end_connectivity']
            # variable in mat file
            mdata_count_pass = mat[atlas + '_count_pass_connectivity']
            mdata_count_end = mat[atlas + '_count_end_connectivity']
            # variable in mat file
            mdata_mean_length_pass = mat[atlas +
                                         '_mean_length_pass_connectivity']
            mdata_mean_length_end = mat[atlas +
                                        '_mean_length_end_connectivity']
            allsub_mat_gfa_pass[i, :, :] = mdata_gfa_pass
            allsub_mat_gfa_end[i, :, :] = mdata_gfa_end
            allsub_mat_volume_weighted_count_pass[i, :, :] = mdata_volume_weighted_count_pass
            allsub_mat_volume_weighted_count_end[i, :, :] = mdata_volume_weighted_count_end
            allsub_mat_count_pass[i, :, :] = mdata_count_pass
            allsub_mat_count_end[i, :, :] = mdata_count_end
            allsub_mat_mean_length_pass[i, :, :] = mdata_mean_length_pass
            allsub_mat_mean_length_end[i, :, :] = mdata_mean_length_end
            i = i + 1
    ii = 0
    for fcon_file in inpath_files_fc:
        fmat = np.loadtxt(fcmatfilespath + fcon_file, delimiter=",")
        print(fcon_file)
        allsub_mat_fcon[ii, :, :] = fmat
        ii = ii + 1

    allsub_mat_gfa_sum = np.add(allsub_mat_gfa_pass, allsub_mat_gfa_end)
    allsub_mat_volume_weighted_count_sum = np.add(
        allsub_mat_volume_weighted_count_pass, allsub_mat_volume_weighted_count_end)
    allsub_mat_count_sum = np.add(allsub_mat_count_pass, allsub_mat_count_end)
    allsub_mat_mean_length_sum = np.add(
        allsub_mat_mean_length_pass, allsub_mat_mean_length_end)

    FC = allsub_mat_fcon  # temporary!!!
    FC[FC < -1] = 0
    FC[FC > 1] = 0
    # perform inverse Fisher's r-to-z transform - Original paper did not use z-transfored FC
    FC = np.arctanh(FC)

    FC_triu = np.zeros((part_num, num_rois, num_rois))
    ii = 0
    for X in list(FC):
        # get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k=0)]
        X[np.triu_indices(X.shape[0], k=0)] = v
        X = X + X.T - np.diag(np.diag(X))
        FC_triu[ii] = X
        ii = ii + 1

    # we need to import the fc matrices in a similar method as above!!! ^
    for edge_weight in ['count_sum', 'volume_weighted_count_sum', 'mean_length_sum', 'gfa_sum']:
        SC = eval('allsub_mat_' + edge_weight)
        SC_triu = np.zeros((part_num, num_rois, num_rois))
        SC_triu_random = np.zeros((part_num, num_rois, num_rois))
        i = 0
        density = np.zeros(part_num)
        for X in list(SC):
            # the following is per suggestion from Elef
            # for each element in X, weight edges by fraction of total connections from that element to other nodes
            for j in np.arange(num_rois):
                for jj in np.arange(num_rois):
                    if X[j, jj] > 0:
                        X[j, jj] = X[j, jj] / np.sum(X[j, :])
            if not no_scramble:
                # scramble X
                X_random = X
                density[i] = bct.density_und(X)[0]
                rng = np.random.default_rng()
                X_random = rng.permuted(X_random, axis=0)
                X_random = rng.permuted(X_random, axis=1)
                v_random = X_random[np.triu_indices(X_random.shape[0], k=0)]
                X_random[np.triu_indices(X_random.shape[0], k=0)] = v_random
                X_random = X_random + X_random.T - np.diag(np.diag(X_random))
                np.fill_diagonal(X_random, 1)
                SC_triu_random[i] = X_random
            # get the upper triangular part of this matrix
            v = X[np.triu_indices(X.shape[0], k=0)]
            X[np.triu_indices(X.shape[0], k=0)] = v
            X = X + X.T - np.diag(np.diag(X))
            # normalize and add diagonal as 1 per Oualid
            # X = X/X.max()
            np.fill_diagonal(X, 1)
            SC_triu[i] = X
            # SC_sparse = csr_matrix(SC_triu[i])
            # SC_triu[i] = SC_sparse.todense()
            i = i + 1

        # Extract the portion of the file name before the first underscore
        file_names = [file.split('_')[0] for file in inpath_files]

        for iii in np.arange(100):
            sc_ids = list(range(len(SC)))
            sc_train_ids, sc_test_ids, fc_train, fc_test, density_train, density_test, file_name_test, file_name_train = train_test_split(
                sc_ids, FC_triu, density, file_names, test_size=0.20, random_state=iii)
            # Write
            f_train = h5py.File('/datain/dataset/train_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_train.create_dataset("inputs", data=SC_triu[sc_train_ids])
            f_train.create_dataset("labels", data=fc_train)
            f_train.create_dataset("density", data=density_train)
            f_train.create_dataset("file_name_train", data=np.array(
                file_name_train, dtype=h5py.string_dtype(encoding='utf-8')))
            f_train.close()
            f_test = h5py.File('/datain/dataset/test_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_test.create_dataset("inputs", data=SC_triu[sc_test_ids])
            f_test.create_dataset("labels", data=fc_test)
            f_test.create_dataset("density", data=density_test)
            f_test.create_dataset("file_name_test", data=np.array(
                file_name_test, dtype=h5py.string_dtype(encoding='utf-8')))
            f_test.close()
            if not no_scramble:
                f_train_scrambled = h5py.File('/datain/dataset/train_scrambled_percent_gqi_' +
                                              edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
                f_train_scrambled.create_dataset("inputs", data=SC_triu_random[sc_train_ids])
                f_train_scrambled.create_dataset("labels", data=fc_train)
                f_train_scrambled.create_dataset("density", data=density_train)
                f_train_scrambled.create_dataset("file_name_train", data=np.array(
                                                 file_name_train, dtype=h5py.string_dtype(encoding='utf-8')))
                f_train_scrambled.close()
                f_test_scrambled = h5py.File('/datain/dataset/test_scrambled_percent_gqi_' +
                                             edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
                f_test_scrambled.create_dataset("inputs", data=SC_triu_random[sc_test_ids])
                f_test_scrambled.create_dataset("labels", data=fc_test)
                f_test_scrambled.create_dataset("density", data=density_test)
                f_test_scrambled.create_dataset("file_name_test", data=np.array(
                                                file_name_test, dtype=h5py.string_dtype(encoding='utf-8')))
                f_test_scrambled.close()
