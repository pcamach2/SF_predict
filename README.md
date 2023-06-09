## README
Python code and associated scripts from resting-state functional connectivity 
prediction from structural connectivity using https://github.com/MICA-MNI/micaopen/tree/master/sf_prediction,
comparing how different structural connectivity matrix edge weightings influence this prediction, and performing
hierachical regression to empirical and predicted resting-state functional connectivity as predictors of clinical outcomes

This code performs the statistical analyses for my thesis (link forthcoming)

### Overall Workflow

* Run python scripts for reading in structural connectivity matrices from different reconstruction workflows and saving out train-test split data
* Run python scripts for reading in train and test data, performing sf_prediction, saving results
* When all predictions finish, run reformat script (`reformat_sf_predict_scores_v3.sh`) in the parent directory of `dataset`
* Move scrambled matrices to a separate folder (`dataset_scrambled`)
* Make train-test split batch results csvs for plotting
* Plot and perform statistical tests

#### Usage

Run python scripts for reading in structural connectivity matrices from different reconstruction workflows and saving out train-test split data
```
test_
```

# Requires container

## TO-DO

* [ ] Create singularity definition file and build instructions
* [ ] Dependencies list in README.md 
* [ ] requirements.txt for python environment
* [ ] Usage examples
