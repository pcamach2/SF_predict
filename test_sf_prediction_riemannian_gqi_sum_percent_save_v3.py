import bct
import h5py
from sklearn.model_selection import train_test_split
import os
import numpy as np
import scipy.io as sio

# struct conn matrices
matfilespath = '/datain/matfiles_gqi_06052023_/'
inpath_files = os.listdir(matfilespath)
part_num = len(inpath_files)

# RSFC matrices
fcmatfilespath = '/datain/matfiles_aroma_06052023/'
inpath_files_fc = os.listdir(fcmatfilespath)
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
    allsub_mat_ncount_pass = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_ncount_end = np.zeros((part_num, num_rois, num_rois))
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
            mdata_ncount_pass = mat[atlas + '_ncount_pass_connectivity']
            mdata_ncount_end = mat[atlas + '_ncount_end_connectivity']
            # variable in mat file
            mdata_count_pass = mat[atlas + '_ncount_pass_connectivity']
            mdata_count_end = mat[atlas + '_ncount_end_connectivity']
            # variable in mat file
            mdata_mean_length_pass = mat[atlas +
                                         '_mean_length_pass_connectivity']
            mdata_mean_length_end = mat[atlas +
                                        '_mean_length_end_connectivity']
            allsub_mat_gfa_pass[i, :, :] = mdata_gfa_pass
            allsub_mat_gfa_end[i, :, :] = mdata_gfa_end
            allsub_mat_ncount_pass[i, :, :] = mdata_ncount_pass
            allsub_mat_ncount_end[i, :, :] = mdata_ncount_end
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
    allsub_mat_ncount_sum = np.add(
        allsub_mat_ncount_pass, allsub_mat_ncount_end)
    allsub_mat_count_sum = np.add(allsub_mat_count_pass, allsub_mat_count_end)
    allsub_mat_mean_length_sum = np.add(
        allsub_mat_mean_length_pass, allsub_mat_mean_length_end)

    FC = allsub_mat_fcon  # temporary!!!
    FC[FC < -1] = 0
    FC[FC > 1] = 0

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
    for edge_weight in ['count_sum', 'ncount_sum', 'mean_length_sum', 'gfa_sum']:
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
            # scramble X
            X_random = X
            density[i] = bct.density_und(X)[0]
            rng = np.random.default_rng()
            X_random = rng.permuted(X_random, axis=0)
            X_random = rng.permuted(X_random, axis=1)
            # get the upper triangular part of this matrix
            v = X[np.triu_indices(X.shape[0], k=0)]
            X[np.triu_indices(X.shape[0], k=0)] = v
            X = X + X.T - np.diag(np.diag(X))
            v_random = X_random[np.triu_indices(X_random.shape[0], k=0)]
            X_random[np.triu_indices(X_random.shape[0], k=0)] = v_random
            X_random = X_random + X_random.T - np.diag(np.diag(X_random))
            # normalize and add diagonal as 1 per Oualid
            # X = X/X.max()
            np.fill_diagonal(X, 1)
            SC_triu[i] = X
            np.fill_diagonal(X_random, 1)
            SC_triu_random[i] = X_random
            # SC_sparse = csr_matrix(SC_triu[i])
            # SC_triu[i] = SC_sparse.todense()
            i = i + 1

        # Extract the portion of the file name before the first underscore
        file_names = [file.split('_')[0] for file in inpath_files]

        for iii in np.arange(100):
            sc_ids = list(range(len(SC)))
            sc_train_ids, sc_test_ids, fc_train, fc_test, density_train, density_test = train_test_split(
                sc_ids, FC_triu, density, test_size=0.20, random_state=iii)
            # Write
            f = h5py.File('/datain/dataset/train_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset("inputs", data=SC_triu[sc_train_ids])
            f.create_dataset("labels", data=fc_train)
            f.create_dataset("density", data=density_train)
            f.create_dataset("file_name", data=np.array(
                file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            f.close()
            f = h5py.File('/datain/dataset/test_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset("inputs", data=SC_triu[sc_test_ids])
            f.create_dataset("labels", data=fc_test)
            f.create_dataset("density", data=density_test)
            f.create_dataset("file_name", data=np.array(
                file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            f.close()
            f = h5py.File('/datain/dataset/train_scrambled_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset("inputs", data=SC_triu_random[sc_train_ids])
            f.create_dataset("labels", data=fc_train)
            f.create_dataset("density", data=density_train)
            f.create_dataset("file_name", data=np.array(
                file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_gqi_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset("inputs", data=SC_triu_random[sc_test_ids])
            f.create_dataset("labels", data=fc_test)
            f.create_dataset("density", data=density_test)
            f.create_dataset("file_name", data=np.array(
                file_names, dtype=h5py.string_dtype(encoding='utf-8')))
            f.close()
