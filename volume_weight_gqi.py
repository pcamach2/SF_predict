# Written by PAUL SHRAP on 10-4-2016
# Weight connectomes by average volume (number of voxels in each ROI)
# Edited for zero volume ROIs by Paul Camacho 01-15-2020, modified for AAL116 atlas 02-08-2023
# Adapted by Paul Camacho 06-21-2023 for GQI connectomes from qsirecon

import os
import csv
from scipy.io import loadmat,savemat
import numpy as np

parcellation_num = 116
# parcellation_num = int(os.environ['parcellation_number'])

# change directory to the directory containing this script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# for all participant IDs in a list, copy the ROI volumes from 
# 	bids/derivatives/structconpipeline_aal116/ResStructConn/<participant ID>/ses-A/Conn116/ROI_Volumes.csv
# 	to the current directory as <participant ID>_ROI_Volumes.csv and save a csv of the connectome from 
#	bids/derivatives/qsirecon/<participant ID>/ses-A/dwi/*gqinetwork.mat
# 	as <participant ID>_conn.csv
# 	Then use the volumes to weight the aal116 count connectome by ROI volume

# read in participant list from txt file
with open('SAY_list_gqi.txt', 'r') as f:
	participant_list = [line.strip() for line in f]

# for each participant, copy ROI_Volumes.csv to current directory
for participant in participant_list:
	ROI_Volumes_file = '/bids/derivatives/structconpipeline_aal116_count_clean/ResStructConn/sub-{}/ses-A/Conn116/ROI_Volumes.csv'.format(participant)
	os.system('cp {} sub-{}_ROI_Volumes.csv'.format(ROI_Volumes_file, participant))
	
# for each participant, load the connectome from the .mat file and save as a csv
for participant in participant_list:
	# connectome_mat_file = '/bids/derivatives/qsirecon/sub-{}/ses-A/dwi/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant, participant)
	# use the mat files from prediction folder
	connectome_mat_file = '/datain/matfiles_gqi_06222023_/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant)
	connectome_mat = loadmat(connectome_mat_file)
	connectome_end = connectome_mat['aal116_count_end_connectivity']
	np.savetxt('{}_gqi_end_conn.csv'.format(participant), connectome_end, delimiter=',')
	connectome_pass = connectome_mat['aal116_count_pass_connectivity']
	np.savetxt('{}_gqi_pass_conn.csv'.format(participant), connectome_pass, delimiter=',')

# for each participant, weight the connectome by ROI volume
for participant in participant_list:
	# load the ROI volumes
	ROI_Volumes_file = 'sub-{}_ROI_Volumes.csv'.format(participant)
	with open(ROI_Volumes_file, 'r') as g:
		reader = csv.reader(g)
		Voxel_nums = [l for l in reader]
	# load the end connectome
	connectome_file = '{}_gqi_end_conn.csv'.format(participant)
	with open(connectome_file, 'r') as g:
		reader = csv.reader(g)
		num_tracts = [l for l in reader]
		# for each ROI, weight volume of ROIs
		for ROI in range(parcellation_num):
			for connection in range(parcellation_num):
				if (float(Voxel_nums[ROI][1])+(float(Voxel_nums[connection][1])))*((float(num_tracts[ROI][connection]))) == 0 :
					num_tracts[ROI][connection]= 0
				else :
					num_tracts[ROI][connection]= (2/(float(Voxel_nums[ROI][1])+(float(Voxel_nums[connection][1])))*((float(num_tracts[ROI][connection]))))
	# save volume weighted end connectome to a csv file
	with open('{}_gqi_end_conn_VolumeWeighted.csv'.format(participant), 'w') as f:
		writer = csv.writer(f)
		writer.writerows(num_tracts)
	# add the volume weighted end connectome to the gqinetwork.mat file
	# connectome_mat_file = '/bids/derivatives/qsirecon/sub-{}/ses-A/dwi/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant, participant)
	connectome_mat_file = '/datain/matfiles_gqi_06222023_/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant)
	connectome_mat = loadmat(connectome_mat_file)
	connectome_mat['aal116_volume_weighted_count_end_connectivity'] = num_tracts
	savemat(connectome_mat_file, connectome_mat)
	# load the pass connectome
	connectome_file = '{}_gqi_pass_conn.csv'.format(participant)
	with open(connectome_file, 'r') as g:
		reader = csv.reader(g)
		num_tracts = [l for l in reader]
		# for each ROI, weight volume of ROIs
		for ROI in range(parcellation_num):
			for connection in range(parcellation_num):
				if (float(Voxel_nums[ROI][1])+(float(Voxel_nums[connection][1])))*((float(num_tracts[ROI][connection]))) == 0 :
					num_tracts[ROI][connection]= 0
				else :
					num_tracts[ROI][connection]= (2/(float(Voxel_nums[ROI][1])+(float(Voxel_nums[connection][1])))*((float(num_tracts[ROI][connection]))))
	# save volume weighted pass connectome to a csv file
	with open('{}_gqi_pass_conn_VolumeWeighted.csv'.format(participant), 'w') as f:
		writer = csv.writer(f)
		writer.writerows(num_tracts)
	# add the volume weighted pass connectome to the gqinetwork.mat file
	# connectome_mat_file = '/bids/derivatives/qsirecon/sub-{}/ses-A/dwi/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant, participant)
	connectome_mat_file = '/datain/matfiles_gqi_06222023_/sub-{}_ses-A_run-1_space-T1w_desc-preproc_gqinetwork.mat'.format(participant)
	connectome_mat = loadmat(connectome_mat_file)
	connectome_mat['aal116_volume_weighted_count_pass_connectivity'] = num_tracts
	savemat(connectome_mat_file, connectome_mat)
