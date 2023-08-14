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

*Scrambled structural connectivity matrices are produced to check whether the structure of the input matrices is 
important for predicting the resting-state functional connectivity. If you are interested in permutation testing, 
a much larger number (100 to 10000) of scrambled matrices per observed ordered matrix must be produced. 
This increases the computational cost and required file storage space by several orders of magnitude. 
If you are not interested in performing such tests, use the option `no_scramble` when running these scripts*

From bash terminal:
``` bash
echo "Saving GQI Training and Testing Splits"
python3 test_sf_prediction_riemannian_gqi_sum_percent_save_v5.py gqi_files_dir fcon_files_dir no_scramble
echo "Saving MSMT CSD Training and Testing Splits"
python3 test_sf_prediction_riemannian_msmt_percent_save_v5.py msmt_files_dir fcon_files_dir no_scramble
echo "Saving DTI Training and Testing Splits"
python3 test_sf_prediction_riemannian_scfsl_percent_save_v5.py scfsl_files_dir fcon_files_dir no_scramble
```

Singularity command:
``` bash
echo "Saving GQI Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/python3 test_sf_prediction_riemannian_gqi_sum_percent_save_v3.py gqi_files_dir fcon_files_dir no_scramble
echo "Saving MSMT CSD Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_save_v3.py msmt_files_dir fcon_files_dir no_scramble
echo "Saving DTI Training and Testing Splits"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_save_v3.py scfsl_files_dir fcon_files_dir no_scramble
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
python3 test_sf_prediction_riemannian_gqi_sum_percent_part1.py no_scramble
python3 test_sf_prediction_riemannian_gqi_sum_percent_part2.py no_scramble
python3 test_sf_prediction_riemannian_gqi_sum_percent_part3.py no_scramble
python3 test_sf_prediction_riemannian_gqi_sum_percent_part4.py no_scramble
echo "Running sf_prediction on MSMT CSD data"
python3 test_sf_prediction_riemannian_msmt_percent_part1.py no_scramble
python3 test_sf_prediction_riemannian_msmt_percent_part2.py no_scramble
python3 test_sf_prediction_riemannian_msmt_percent_part3.py no_scramble
python3 test_sf_prediction_riemannian_msmt_percent_part4.py no_scramble
echo "Running sf_prediction on DTI data"
python3 test_sf_prediction_riemannian_scfsl_percent_part1.py no_scramble
python3 test_sf_prediction_riemannian_scfsl_percent_part2.py no_scramble
python3 test_sf_prediction_riemannian_scfsl_percent_part3.py no_scramble
python3 test_sf_prediction_riemannian_scfsl_percent_part4.py no_scramble
```


Singularity command:
``` bash
echo "Running sf_prediction on GQI data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part1.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part2.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part3.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_gqi_sum_percent_part4.py no_scramble
echo "Running sf_prediction on MSMT CSD data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part1.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part2.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part3.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_msmt_percent_part4.py no_scramble
echo "Running sf_prediction on DTI data"
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part1.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part2.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part3.py no_scramble
singularity exec -B ./:/datain pyconnpredict-v1.0.0.sif python3 /datain/test_sf_prediction_riemannian_scfsl_percent_part4.py no_scramble
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

##### Move scrambled matrices to a separate folder (`dataset_scrambled`) *optional*

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

#### Clinical data prediction

The script used here can be adjusted for four main options:
* Train-test split batch number
* Reconstruction + structural connectivity matrix weight
* Cardiorespiratory fitness and clinical motor function tests to include in factor analysis to produce functional fitness score
* Specific demographic data to use as predictors

Ensure that your demographic and clinical data is complete and contained in a csv file in the prediction folder
*Include participant ID numbers to match with training and testing matrices created above*

##### Functional Fitness Score & RSFC PCA Regression

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
