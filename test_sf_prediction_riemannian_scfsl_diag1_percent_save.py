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

os.chdir('/datain')
#struct conn matrices
matfilespath = '/datain/matfiles_dti_volwei_11182022_/' # '/datain/matfiles_dti_count_11182022_/'
inpath_files = os.listdir(matfilespath)
part_num = len(inpath_files)

# RSFC matrices

fcmatfilespath = '/datain/matfiles_aroma_11182022/'
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
# global_efficiency_mat = np.zeros((part_num,len(atlases)))
# # atlas_scores = {}
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
    # elif atlas == 'power264':
    #     labels = power264_keys['roi_id']
    #     keys = power264_keys
    #     num_rois = 264
    # elif atlas == 'gordon333':
    #     labels = gordon333_keys['roi_id']
    #     keys = gordon333_keys
    #     num_rois = 333
    # elif atlas == 'aicha384':
    #     labels = aicha384_keys['roi_id']
    #     keys = aicha384_keys
    #     num_rois = 384
    # elif atlas == 'brainnetome246':
    #     labels = brainnetome246_keys['roi_id']
    #     keys = brainnetome246_keys
    #     num_rois = 246
    # elif atlas == 'schaefer100x17':
    #     labels = schaefer100x17_keys['roi_id']
    #     keys = schaefer100x17_keys
    #     num_rois = 100
    # elif atlas == 'schaefer100x7':
    #     labels = schaefer100x7_keys['roi_id']
    #     keys = schaefer100x7_keys
    #     num_rois = 100
    # elif atlas == 'schaefer200x17':
    #     labels = schaefer200x17_keys['roi_id']
    #     keys = schaefer200x17_keys
    #     num_rois = 200
    # elif atlas == 'schaefer200x7':
    #     labels = schaefer200x7_keys['roi_id']
    #     keys = schaefer200x7_keys
    #     num_rois = 200
    # elif atlas == 'schaefer400x17':
    #     labels = schaefer400x17_keys['roi_id']
    #     keys = schaefer400x17_keys
    #     num_rois = 400
    # elif atlas == 'schaefer400x7':
    #     labels = schaefer400x7_keys['roi_id']
    #     keys = schaefer400x7_keys
    #     num_rois = 400
    print("Concatenating all structural connectomes for %s parcellation" % atlas)
    # GQI
    # nodal
    # connectome
    # allsub_mat_gfa_pass = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_gfa_end = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_ncount_pass = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_ncount_end = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_count_pass = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_count_end = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_mean_length_pass = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_mean_length_end = np.zeros((part_num,num_rois,num_rois))
    # CSD + SIFT2
    # connectome
    # allsub_mat_sift_radius2_count = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_sift_invnodevol_radius2_count = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_radius2_count = np.zeros((part_num,num_rois,num_rois))
    # allsub_mat_radius2_meanlength = np.zeros((part_num,num_rois,num_rois))
    allsub_mat_fcon = np.zeros((part_num,num_rois,num_rois))
    allsub_mat_dti_volumeweighted = np.zeros((part_num,num_rois,num_rois))
    allsub_mat_dti_meanlength = np.zeros((part_num,num_rois,num_rois))
    allsub_mat_dti_count = np.zeros((part_num,num_rois,num_rois))

    i = 0
    for matfile in inpath_files:
        #mat=sio.loadmat(matfilespath + matfile)# load mat-file
        mat = np.genfromtxt(matfilespath + matfile, delimiter=",") # needs to read in csv
        print(matfile)
        if 'msmtconnectome' in matfile:
            mdata_sift_radius2_count = mat[atlas+ '_sift_radius2_count_connectivity']  # variable in mat file
            mdata_sift_invnodevol_radius2_count = mat[atlas+'_sift_invnodevol_radius2_count_connectivity']  # variable in mat file
            mdata_radius2_count = mat[atlas+'_radius2_count_connectivity']  # variable in mat file
            mdata_radius2_meanlength = mat[atlas+'_radius2_meanlength_connectivity']  # variable in mat file
            # set 3d matrices
            allsub_mat_sift_radius2_count[i,:,:] = mdata_sift_radius2_count
            allsub_mat_sift_invnodevol_radius2_count[i,:,:] = mdata_sift_invnodevol_radius2_count
            allsub_mat_radius2_count[i,:,:] = mdata_radius2_count
            allsub_mat_radius2_meanlength[i,:,:] = mdata_radius2_meanlength
            i = i + 1
        elif 'gqi' in matfile:
            mdata_gfa_pass = mat[atlas + '_gfa_pass_connectivity']  # variable in mat file
            mdata_gfa_end = mat[atlas + '_gfa_end_connectivity'] 
            mdata_ncount_pass = mat[atlas + '_ncount_pass_connectivity']  # variable in mat file
            mdata_ncount_end = mat[atlas + '_ncount_end_connectivity'] 
            mdata_count_pass = mat[atlas + '_ncount_pass_connectivity']  # variable in mat file
            mdata_count_end = mat[atlas + '_ncount_end_connectivity']
            mdata_mean_length_pass = mat[atlas + '_mean_length_pass_connectivity']  # variable in mat file
            mdata_mean_length_end = mat[atlas + '_mean_length_end_connectivity']
            allsub_mat_gfa_pass[i,:,:] = mdata_gfa_pass
            allsub_mat_gfa_end[i,:,:] = mdata_gfa_end
            allsub_mat_ncount_pass[i,:,:] = mdata_ncount_pass
            allsub_mat_ncount_end[i,:,:] = mdata_ncount_end
            allsub_mat_count_pass[i,:,:] = mdata_count_pass
            allsub_mat_count_end[i,:,:] = mdata_count_end
            allsub_mat_mean_length_pass[i,:,:] = mdata_mean_length_pass
            allsub_mat_mean_length_end[i,:,:] = mdata_mean_length_end
            i = i + 1
        elif 'dti' in matfile:
            # mdata_dti_volwei = mat  # variable in mat file
            if 'Count' in matfile:
                allsub_mat_dti_count[i,:,:] = mat
            elif 'Volume' in matfile:
                allsub_mat_dti_volumeweighted[i,:,:] = mat
            elif 'mean' in matfile:
                # mdata_dti_meanlength = mat
                allsub_mat_dti_meanlength[i,:,:] = mat
            i = i + 1

    ii = 0
    for fcon_file in inpath_files_fc:
        fmat = np.loadtxt(fcmatfilespath + fcon_file, delimiter=",")
        print(fcon_file)
        allsub_mat_fcon[ii,:,:] = fmat
        ii = ii + 1
    
    FC = allsub_mat_fcon # temporary!!!
    FC[FC<-1]=0
    FC[FC>1]=0
    
    FC_triu = np.zeros((part_num,num_rois,num_rois))
    ii = 0
    for X in list(FC):
        #get the upper triangular part of this matrix
        v = X[np.triu_indices(X.shape[0], k = 0)]
        X[np.triu_indices(X.shape[0], k = 0)] = v
        X = X + X.T - np.diag(np.diag(X))
        FC_triu[ii] = X
        ii = ii + 1

    # we need to import the fc matrices in a similar method as above!!! ^
#    for edge_weight in 'count_pass','count_end','ncount_pass','ncount_end','mean_length_pass','mean_length_end':
    for edge_weight in 'dti_volumeweighted', 'dti_count', 'dti_meanlength':
    # if num_rois == 116:    
        # edge_weight = 'dti_count','dti_volumeweighted','' # "dti_count"
        SC = eval('allsub_mat_' + edge_weight)
        SC_triu = np.zeros((part_num,num_rois,num_rois))
        SC_triu_random = np.zeros((part_num,num_rois,num_rois))
        i = 0
        for X in list(SC):
            # the folowing is per suggestion from Elef
            # for each element in X, weight edges by fraction of total connections from that element to other nodes
            for j in np.arange(num_rois):
                for jj in np.arange(num_rois):
                    if X[j,jj] > 0:
                        X[j,jj] = X[j,jj]/np.sum(X[j,:])
            # scramble X
            X_random = X
            rng = np.random.default_rng()
            X_random = rng.permuted(X_random, axis=0)
            X_random = rng.permuted(X_random, axis=1)
            # get the upper triangular part of this matrix
            v = X[np.triu_indices(X.shape[0], k = 0)]
            X[np.triu_indices(X.shape[0], k = 0)] = v
            X = X + X.T - np.diag(np.diag(X))
            v_random = X_random[np.triu_indices(X_random.shape[0], k = 0)]
            X_random[np.triu_indices(X_random.shape[0], k = 0)] = v_random
            X_random = X_random + X_random.T - np.diag(np.diag(X_random))
            # normalize and add diagonal as 1 per Oualid
            # X = X/X.max()
            np.fill_diagonal(X, 1)
            SC_triu[i] = X 
            np.fill_diagonal(X_random, 1)
            SC_triu_random[i] = X_random
            #SC_sparse = csr_matrix(SC_triu[i])
            #SC_triu[i] = SC_sparse.todense()
            i = i + 1

        for iii in np.arange(100):
            sc_ids = list(range(len(SC)))
            sc_train_ids, sc_test_ids, fc_train, fc_test = train_test_split(sc_ids, FC_triu, test_size = 0.20, random_state=iii)
            # Write
            f = h5py.File('/datain/dataset/train_percent_dti_' + edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset(f"inputs", data=SC_triu[sc_train_ids])
            f.create_dataset(f"labels", data=fc_train)
            f.close()
            f = h5py.File('/datain/dataset/train_scrambled_percent_dti_' + edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset(f"inputs", data=SC_triu_random[sc_train_ids])
            f.create_dataset(f"labels", data=fc_train)
            f.close()
            f = h5py.File('/datain/dataset/test_percent_dti_' + edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset(f"inputs", data=SC_triu[sc_test_ids])
            f.create_dataset(f"labels", data=fc_test)
            f.close()
            f = h5py.File('/datain/dataset/test_scrambled_percent_dti_' + edge_weight + '_diag1_' + str(iii) + '.h5py', 'w')
            f.create_dataset(f"inputs", data=SC_triu_random[sc_test_ids])
            f.create_dataset(f"labels", data=fc_test)
            f.close()

