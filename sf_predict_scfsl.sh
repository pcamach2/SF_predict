#!/bin/bash
#
#SBATCH --job-name=sf_predict_SAY_scfsl_v2
#SBATCH --output=sf_predict_SAY_scfsl_v2.txt
#SBATCH --ntasks-per-node=1
#SBATCH --time=128:00:00
#
# Usage:
#
#       # Make and save training and testing splits:
#       sbatch --a 01 sf_predict_scfsl.sh /path/to/base/directory/ beta no project_directory_name 
#       # Load saved training and testing splits and run prediction:
#       sbatch --a 01,02,03,04 sf_predict_scfsl.sh /path/to/base/directory/ beta yes project_directory_name

base_dir=$1
version=$2
save=$3
scripts=${base_dir}/${version}/scripts
prediction_dir=$4

NOW=$(date "+%D-%T")
if [ "${save}" == "yes" ];
then
	singularity exec -B ${base_dir}/${version}/testing/${prediction_dir}:/datain ${base_dir}/singularity_images/pyconnpredict_0.2.6.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_save_v5.py dti_files fcon_files no_scramble
	exit 0

elif [ "${save}" == "no" ];
then
	if [ "${SLURM_ARRAY_TASK_ID}" == "1" ];
	then
	    singularity exec -B ${base_dir}/${version}/testing/${prediction_dir}:/datain ${base_dir}/singularity_images/pyconnpredict_0.2.6.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part1.py no_scramble
    elif [ "${SLURM_ARRAY_TASK_ID}" == "2" ];
	then
		singularity exec -B ${base_dir}/${version}/testing/${prediction_dir}:/datain ${base_dir}/singularity_images/pyconnpredict_0.2.6.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part2.py no_scramble
    elif [ "${SLURM_ARRAY_TASK_ID}" == "3" ];	
	then	
		singularity exec -B ${base_dir}/${version}/testing/${prediction_dir}:/datain ${base_dir}/singularity_images/pyconnpredict_0.2.6.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part3.py no_scramble
    elif [ "${SLURM_ARRAY_TASK_ID}" == "4" ];
	then
		singularity exec -B ${base_dir}/${version}/testing/${prediction_dir}:/datain ${base_dir}/singularity_images/pyconnpredict_0.2.6.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part4.py no_scramble
	fi
        exit 0
fi
