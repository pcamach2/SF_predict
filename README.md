## README
Python code and associated scripts for resting-state functional connectivity 
prediction from structural connectivity using https://github.com/MICA-MNI/micaopen/tree/master/sf_prediction,
comparing how different structural connectivity matrix edge weightings influence this prediction, and performing
hierachical regression to empirical and predicted resting-state functional connectivity as predictors of clinical outcomes

This code performs the statistical analyses for my thesis (link forthcoming)

Structural connectomes used here were produced using QSIPrep as part of
the BIC 3T MRI Processing Pipeline https://github.com/mrfil/pipeline-hpc/tree/SAY
and custom DTI-based structural connectivity pipeline (https://github.com/mrfil/scfsl), 
which uses QSIPrep preprocessed derivatives as inputs for the CUDA-accelerated `bedpostx_gpu` and `probtrackx2_gpu` 

All connectomes should be moved to a subfolder of the directory containing these scripts.

### Overall Workflow

* (Prerequisite) Process T1w, DWI, and resting-state fMRI using the [BIC MRI Pipeline](https://github.com/mrfil/pipeline-hpc/tree/SAY)
* Use ROI volumes from SCFSL to weight the streamline count matrix from DSI Studio GQI workflow, save results in `gqinetwork.mat` files with the `volume_weight_gqi.py` script
* Run python scripts for reading in structural connectivity matrices from different reconstruction workflows and saving out train-test split data
* Run python scripts for reading in train and test data, performing sf_prediction, saving results
* When all predictions finish, run reformat script (`reformat_sf_predict_scores_v3.sh`) in the parent directory of `dataset`
* Move scrambled matrices to a separate folder (`dataset_scrambled`)
* Make train-test split batch results csvs for plotting
* Plot and perform statistical tests

#### Usage

##### Use ROI volumes from SCFSL to weight the streamline count matrix from DSI Studio GQI workflow, save results in `gqinetwork.mat` files with the `volume_weight_gqi.py` script

From bash terminal:
``` bash
echo "Weighting streamline count matrices from GQI using ROI Volumes"
python3 volume_weight_gqi.py
```

Singularity command:
``` bash
echo "Saving GQI Training and Testing Splits"
singularity exec -B ./:/datain,../PROJECT/bids:/bids pyconnpredict-v1.0.0.sif python3 /datain/volume_weight_gqi.py
```

##### Run python scripts for reading in structural connectivity matrices from different reconstruction workflows and saving out train-test split data

From bash terminal:
``` bash
echo "Saving GQI Training and Testing Splits"
python3 test_sf_prediction_riemannian_gqi_sum_percent_save_v3.py
echo "Saving MSMT CSD Training and Testing Splits"
python3 test_sf_prediction_riemannian_msmt_percent_save_v3.py
echo "Saving DTI Training and Testing Splits"
python3 test_sf_prediction_riemannian_scfsl_percent_save_v3.py
```

Singularity command:
``` bash
echo "Saving GQI Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/python3 test_sf_prediction_riemannian_gqi_sum_percent_save_v3.py
echo "Saving MSMT CSD Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_save_v3.py
echo "Saving DTI Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_save_v3.py
```

Using slurm:
```bash
sbatch -a 01 sf_predict_gqi.sh /path/to/base/directory/ beta yes project_directory_name
sbatch -a 01 sf_predict_msmt.sh /path/to/base/directory/ beta yes project_directory_name
sbatch -a 01 sf_predict_scfsl.sh /path/to/base/directory/ beta yes project_directory_name
```

##### Run python scripts for reading in train and test data, performing sf_prediction, saving results

From bash terminal:
```bash
echo "Running sf_prediction on GQI data"
python3 test_sf_prediction_riemannian_gqi_sum_percent_part1.py
python3 test_sf_prediction_riemannian_gqi_sum_percent_part2.py
python3 test_sf_prediction_riemannian_gqi_sum_percent_part3.py
python3 test_sf_prediction_riemannian_gqi_sum_percent_part4.py
echo "Running sf_prediction on MSMT CSD data"
python3 test_sf_prediction_riemannian_msmt_percent_part1.py
python3 test_sf_prediction_riemannian_msmt_percent_part2.py
python3 test_sf_prediction_riemannian_msmt_percent_part3.py
python3 test_sf_prediction_riemannian_msmt_percent_part4.py
echo "Running sf_prediction on DTI data"
python3 test_sf_prediction_riemannian_scfsl_percent_part1.py
python3 test_sf_prediction_riemannian_scfsl_percent_part2.py
python3 test_sf_prediction_riemannian_scfsl_percent_part3.py
python3 test_sf_prediction_riemannian_scfsl_percent_part4.py
```


Singularity command:
``` bash
echo "Running sf_prediction on GQI data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part1.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part2.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part3.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part4.py
echo "Running sf_prediction on MSMT CSD data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part1.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part2.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part3.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part4.py
echo "Running sf_prediction on DTI data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part1.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part2.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part3.py
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part4.py
```

Using slurm:
```bash
sbatch --a 01,02,03,04 sf_predict_gqi.sh /path/to/base/directory/ beta yes project_directory_name
sbatch --a 01,02,03,04 sf_predict_msmt.sh /path/to/base/directory/ beta yes project_directory_name
sbatch --a 01,02,03,04 sf_predict_scfsl.sh /path/to/base/directory/ beta yes project_directory_name
```

##### When all predictions finish, run reformat script (`reformat_sf_predict_scores_v3.sh`) in the parent directory of `dataset`

From bash terminal:
```bash
reformat_sf_predict_scores_v5.sh
```

##### Move scrambled matrices to a separate folder (`dataset_scrambled`)

From bash terminal:
```bash
mkdir ./dataset_scrambled
mv ./dataset/*scrambled* ./dataset_scrambled
```

##### Make train-test split batch results csvs for plotting

From bash terminal:
``` bash
python3 collect_scores_percent.py
```

Singularity command:
``` bash
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/collect_scores_percent.py
```

##### Plot and perform statistical tests using the included Jupyter notebook

From bash terminal:
``` bash
jupyter-notebook plotting_and_stats.ipynb
```

Singularity command to run python script:
``` bash
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/plotting_and_stats.py
```

#### Clinical data prediction - RSFC PCA Regression of Functional Fitness Score 

From bash terminal:
``` bash
python3 functional_fitness_modeling.py /path/to/prediction_folder
```

Singularity command to run python script:
``` bash
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/functional_fitness_modeling.py /path/to/prediction_folder
```

### Requirements

This project utilizes a model from the following paper, which should be cited:

Benkarim O, Paquola C, Park B, Royer J, Rodríguez-Cruces R, Vos de Wael R, Misic B, Piella G, Bernhardt B. (2022) A Riemannian approach to predicting brain function from the structural connectome. NeuroImage 257, 119299. https://doi.org/10.1016/j.neuroimage.2022.119299.

Other requirements:
* [Python Packages](requirements.txt)
* [Jupyter Notebook](https://jupyter.org/install) for plotting and statistical tests
* Bash terminal for shell scripts

A [Dockerfile](docker/Dockerfile) is included for building Docker (and subsequent Singularity) images if a containerized implementation is required.

## TO-DO - RSFC Predict From SC

* [X] Create Dockerfile and build instructions for Docker and Singularity
* [X] Singularity examples
* [X] Dependencies list in README.md 
* [X] requirements.txt for python environment
* [X] Usage examples
* [X] Slurm scripts
* [X] Slurm examples
* [X] Create and streamline Jupyter notebook for plotting and stats
* [X] Make python script as alternative to Jupyter notebook
* [X] Remove unused python scripts
* [X] Clarify intended use is with fraction weighting

## TO-DO - Clinical Outcome Prediction

* [X] Create Dockerfile and build instructions for Docker and Singularity
* [X] Singularity examples
* [X] Dependencies list in README.md 
* [X] requirements.txt for python environment
* [X] Usage examples
* [ ] Rename files (remove diag1) -optional-
* [X] Remove unused python scripts
