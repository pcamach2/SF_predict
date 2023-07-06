import os
import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import bct

os.chdir('/datain')

# struct conn matrices
matfilespath = '/datain/matfiles_dti_volwei_06222023_/'
inpath_files = sorted(os.listdir(matfilespath))
part_num = len(inpath_files)

# RSFC matrices
fcmatfilespath = '/datain/matfiles_aroma_06222023_/'
inpath_files_fc = sorted(os.listdir(fcmatfilespath))
part_num_fc = len(inpath_files_fc)

scripts = '/datain/atlas_ids/'
# ,'schaefer100x17', 'schaefer100x17','schaefer100x7','schaefer200x17','schaefer200x7','schaefer400x17','schaefer400x7'
atlases = ['aal116']
# ,'power264','gordon333','aicha384','brainnetome246'

for atlas in atlases:
    labels = []
    keys = []
    num_rois = 0
    if atlas == 'aal116':
        # labels = aal116_keys['roi_id']
        # keys = aal116_keys
        num_rois = 116
    print("Concatenating all structural connectomes for %s parcellation" % atlas)
    allsub_mat_fcon = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_dti_volumeweighted = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_dti_meanlength = np.zeros((part_num, num_rois, num_rois))
    allsub_mat_dti_count = np.zeros((part_num, num_rois, num_rois))

    i = 0
    for matfile in inpath_files:
        mat = np.genfromtxt(matfilespath + matfile,
                            delimiter=",", dtype='float', encoding='us-ascii')
        print(matfile)
        if 'dti' in matfile:
            # mdata_dti_volwei = mat  # variable in mat file
            if 'Count' in matfile:
                allsub_mat_dti_count[i, :, :] = mat
            elif 'Volume' in matfile:
                allsub_mat_dti_volumeweighted[i, :, :] = mat
            elif 'mean' in matfile:
                allsub_mat_dti_meanlength[i, :, :] = mat
            i = i + 1

    ii = 0
    for fcon_file in inpath_files_fc:
        fmat = np.loadtxt(fcmatfilespath + fcon_file, delimiter=",")
        print(fcon_file)
        allsub_mat_fcon[ii, :, :] = fmat
        ii = ii + 1

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

    # for edge_weight in ['dti_volumeweighted', 'dti_count', 'dti_meanlength']:
    if num_rois == 116:
        edge_weight = 'dti_volumeweighted'
        SC = eval('allsub_mat_' + edge_weight)
        SC_triu = np.zeros((part_num, num_rois, num_rois))
        SC_triu_random = np.zeros((part_num, num_rois, num_rois))
        density = np.zeros(part_num)
        i = 0
        for X in list(SC):
            # the following is per suggestion from Elef
            # for each element in X, weight edges by fraction of total connections from that element to other nodes
            for j in np.arange(num_rois):
                for jj in np.arange(num_rois):
                    if X[j, jj] > 0:
                        X[j, jj] = X[j, jj] / np.sum(X[j, :])
            density[i] = bct.density_und(X)[0]
            # scramble X
            X_random = X
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
            sc_train_ids, sc_test_ids, fc_train, fc_test, density_train, density_test, file_name_test, file_name_train = train_test_split(
                sc_ids, FC_triu, density, file_names, test_size=0.20, random_state=iii)
            # Write
            f_train = h5py.File('/datain/dataset/train_percent_dti_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_train.create_dataset("inputs", data=SC_triu[sc_train_ids])
            f_train.create_dataset("labels", data=fc_train)
            f_train.create_dataset("density", data=density_train)
            f_train.create_dataset("file_name_train", data=np.array(
                file_name_train, dtype=h5py.string_dtype(encoding='utf-8')))
            f_train.close()
            f_test = h5py.File('/datain/dataset/test_percent_dti_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_test.create_dataset("inputs", data=SC_triu[sc_test_ids])
            f_test.create_dataset("labels", data=fc_test)
            f_test.create_dataset("density", data=density_test)
            f_test.create_dataset("file_name_test", data=np.array(
                file_name_test, dtype=h5py.string_dtype(encoding='utf-8')))
            f_test.close()
            f_train_scrambled = h5py.File('/datain/dataset/train_scrambled_percent_dti_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_train_scrambled.create_dataset("inputs", data=SC_triu_random[sc_train_ids])
            f_train_scrambled.create_dataset("labels", data=fc_train)
            f_train_scrambled.create_dataset("density", data=density_train)
            f_train_scrambled.create_dataset("file_name_train", data=np.array(
                file_name_train, dtype=h5py.string_dtype(encoding='utf-8')))
            f_train_scrambled.close()
            f_test_scrambled = h5py.File('/datain/dataset/test_scrambled_percent_dti_' +
                          edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f_test_scrambled.create_dataset("inputs", data=SC_triu_random[sc_test_ids])
            f_test_scrambled.create_dataset("labels", data=fc_test)
            f_test_scrambled.create_dataset("density", data=density_test)
            f_test_scrambled.create_dataset("file_name_test", data=np.array(
                file_name_test, dtype=h5py.string_dtype(encoding='utf-8')))
            f_test_scrambled.close()
