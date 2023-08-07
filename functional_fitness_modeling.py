# functional_fitness_modeling.py [path to sf_prediction folder]
#
# Read in empirical and predicted functional connectivity matrices and clinical outcomes data
# perform generalized least squares regression to predict functional fitness score (from factor analysis performed here)
# from empirical functional connectivity, predicted functional connectivity, and demographic data
#
# Written by: Paul B Camacho (pcamach2@illinois.edu)

# importing packages
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
import statsmodels.api as sm
import sys
import datetime
from tabulate import tabulate


def get_indices_first_n_pca(X, n_components=50):
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Get the indices of the first fifty components
    indices = np.argwhere(
        np.sum(pca.components_[:n_components], axis=0) != 0).flatten()

    # Create a DataFrame with component weightings and explained variance
    components_df = pd.DataFrame({
        'Component': range(1, pca.n_components_ + 1),
        'Weightings': np.sum(pca.components_[:pca.n_components_], axis=1),
        'Variance Explained': pca.explained_variance_ratio_
    })

    # Save DataFrame to a CSV file
    components_df.to_csv('pca_statistics.csv', index=False)

    # Plot PCA statistics
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Component', y='Weightings', data=components_df)
    plt.title('Component Weightings')
    plt.xlabel('Component')
    plt.ylabel('Weightings')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Component', y='Variance Explained', data=components_df)
    plt.title('Variance Explained')
    plt.xlabel('Component')
    plt.ylabel('Variance Explained')
    plt.show()

    return indices


# function to use scikit-learn's test_train_split function to split a behavioral data df into train and test sets
def split_data(df, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    # split data into train and test sets
    train, test = train_test_split(
        df, test_size=test_size, random_state=random_state)
    return train, test


def get_principal_components(original_matrix, new_matrix, num_components=50):
    # Scale and preprocess the data
    scaler = StandardScaler()
    original_matrix_scaled = scaler.fit_transform(original_matrix)
    new_matrix_scaled = scaler.transform(new_matrix)

    # Perform PCA on original matrix
    pca = PCA(n_components=num_components)
    pca.fit(original_matrix_scaled)

    # Weightings from PCA
    weightings = pca.components_

    # Principal components of original matrix
    original_principal_components = np.dot(
        original_matrix_scaled, weightings.T)

    # Principal components of new matrix
    new_principal_components = np.dot(new_matrix_scaled, weightings.T)

    # perform PCA on the new matrix, not influenced by the original matrix
    pca_new = PCA(n_components=num_components)
    pca_new.fit(new_matrix_scaled)

    # Weightings from PCA
    weightings_new = pca_new.components_

    # Principal components of new matrix
    new_principal_components_ = np.dot(new_matrix_scaled, weightings_new.T)

    return original_principal_components, new_principal_components, new_principal_components_


# function to read in the predicted functional connectivity matrices and the corresponding file_names,
# return a dataframe with the predicted functional connectivity matrices and the corresponding file_names
def get_predicted_FC_with_labels(prediction_path, edge_weight, batch_num):
    # add / to end of prediction_path if not already there
    if prediction_path[-1] != '/':
        prediction_path = prediction_path + '/'
    # create empty array to store predicted functional connectivity matrices and the corresponding file_names
    with open(prediction_path + 'preds_percent_' + atlas + '_' + edge_weight + '_diag1_n' + str(batch_num) + '.pkl', 'rb') as f_pred:
        preds = pickle.load(f_pred)
    print(preds[9].keys())
    # get the predicted functional connectivity matrices from walk length 4
    pred_FC_test = preds[4]['test']
    pred_FC_train = preds[4]['train']
    # get the corresponding file_names from h5py
    file_names_test = h5py.File(prediction_path + 'test_percent_' +
                                edge_weight + '_diag1_' + str(batch_num) + '.h5py', 'r')
    file_names_test = file_names_test['file_name_test'][:]
    file_names_test = [file_name.decode('utf-8')
                       for file_name in file_names_test]
    # get the corresponding file_names from h5py
    file_names_train = h5py.File(prediction_path + 'train_percent_' +
                                 edge_weight + '_diag1_' + str(batch_num) + '.h5py', 'r')
    file_names_train = file_names_train['file_name_train'][:]
    file_names_train = [file_name.decode('utf-8')
                        for file_name in file_names_train]
    # combine the file_names from test and training sets to an array of length = train + test
    file_names = np.hstack((file_names_test, file_names_train))
    # combine the predicted functional connectivity matrices from test and training sets
    pred_FC = np.vstack((pred_FC_test, pred_FC_train))
    # convert pred_FC_test_with_labels and pred_FC_train_with_labels to pandas dataframes
    pred_FC_test_with_labels = pd.DataFrame(
        pred_FC_test)
    pred_FC_train_with_labels = pd.DataFrame(
        pred_FC_train)
    #  add a column to the predicted functional connectivity dataframes to indicate 
    #  whether subject is in the training set (test0_train1=1) or test set (test0_train1=0)
    pred_FC_test_with_labels['test0_train1'] = 0
    pred_FC_train_with_labels['test0_train1'] = 1
    # combine predicted FC dataframes from test and training
    pred_FC_with_labels = pd.concat(
        [pred_FC_test_with_labels, pred_FC_train_with_labels], axis=0)
    # rename columns to indicate edge number, unless column is file_name or test0_train1
    for column in pred_FC_with_labels.columns:
        if column != 'file_name_train' and column != 'file_name_test' and column != 'test0_train1':
            pred_FC_with_labels = pred_FC_with_labels.rename(
                columns={column: 'predicted_' + str(column)})
    # return the dataframe
    return pred_FC_with_labels


# function to read in the empirical functional connectivity matrices and the corresponding file_names from h5py,
#     return a dataframe with the empirical functional connectivity matrices and the corresponding file_names
def get_empirical_FC_with_labels(prediction_path, edge_weight, batch_num):
    # add / to end of prediction_path if not already there
    if prediction_path[-1] != '/':
        prediction_path = prediction_path + '/'
    # get the empirical functional connectivity matrices and file_names from test h5py as dictionaries
    test_FC_h5py = h5py.File(prediction_path + 'test_percent_' +
                                  edge_weight + '_diag1_' + str(batch_num) + '.h5py', 'r')
    test_FC = list(np.array(test_FC_h5py.get('labels')))
    print(len(test_FC))
    # get flattened upper triangle of fc_test matrices for each subject
    for ii in range(len(test_FC)):
        test_FC[ii] = test_FC[ii][np.triu_indices(116, k=1)]
    test_file_names = list(np.array(test_FC_h5py.get('file_name_test')))
    test_file_names = [file_name.decode('utf-8')
                          for file_name in test_file_names]
    # combine lists into a dictionary, preserving order of elements in both lists
    test_FC_dict = dict(zip(test_file_names, test_FC))
    # get the empirical functional connectivity matrices and file_names from train h5py as dictionaries
    train_FC_h5py = h5py.File(prediction_path + 'train_percent_' +
                                      edge_weight + '_diag1_' + str(batch_num) + '.h5py', 'r')
    train_FC = list(np.array(train_FC_h5py.get('labels')))
    print(len(train_FC))
    # get flattened upper triangle of fc_train matrices for each subject
    for ii in range(len(train_FC)):
        train_FC[ii] = train_FC[ii][np.triu_indices(116, k=1)]
    # print(train_FC)
    train_file_names = list(np.array(train_FC_h5py.get('file_name_train')))
    train_file_names = [file_name.decode('utf-8')
                            for file_name in train_file_names]
    # combine lists into a dictionaryedg
    train_FC_dict = dict(zip(train_file_names, train_FC)) 
    # convert the dictionaries to pandas dataframes
    test_FC_with_labels = pd.DataFrame.from_dict(test_FC_dict, orient='index')
    test_FC_with_labels.reset_index(inplace=True)
    test_FC_with_labels.rename(columns={'index': 'file_name'}, inplace=True)
    train_FC_with_labels = pd.DataFrame.from_dict(train_FC_dict, orient='index')
    train_FC_with_labels.reset_index(inplace=True)
    train_FC_with_labels.rename(columns={'index': 'file_name'}, inplace=True)
    # add a column to the empirical functional connectivity dataframe to indicate whether the subject is in the training set (test0_train1=1) or test set (test0_train1=0)
    test_FC_with_labels['test0_train1'] = np.zeros(test_FC_with_labels.shape[0])
    train_FC_with_labels['test0_train1'] = np.ones(train_FC_with_labels.shape[0])
    # combine the test and training dataframes
    FC_with_labels = pd.concat([test_FC_with_labels, train_FC_with_labels])
    print(FC_with_labels.shape)
    # clean file_name column to only "sub-SAY###", where "###" is the subject number
    FC_with_labels['file_name'] = FC_with_labels['file_name'].str.replace('_ses-A_dwi', '')
    # print(FC_with_labels.head())
    return FC_with_labels

# function to plot predicted vs. empirical functional connectivity matrices
def plot_predicted_vs_empirical_FC(predicted_FC, empirical_FC, edge_weight, batch_num, atlas, nr=None):
    num_subjects = predicted_FC.shape[0]
    num_rois = 116
    if atlas != 'aal116' and nr is not None:
        num_rois = nr
    # convert the predicted functional connectivity vector of upper triangle values to an NxN matrix
    predicted_FC_matrix = np.zeros((num_rois, num_rois))
    predicted_FC_matrix[np.triu_indices(num_rois, k=1)] = predicted_FC
    predicted_FC_matrix = predicted_FC_matrix + predicted_FC_matrix.T
    # plot the predicted functional connectivity matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(predicted_FC_matrix, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Predicted FC matrix, ' + edge_weight + ', batch ' + str(batch_num))
    plt.savefig(predict_path + 'plots/' + 'predicted_FC_matrix_' + edge_weight + '_batch_' + str(batch_num) + '.png')
    plt.close()
    # convert the empirical functional connectivity vector of upper triangle values to an NxN matrix
    empirical_FC_matrix = np.zeros((num_rois, num_rois))
    empirical_FC_matrix[np.triu_indices(num_rois, k=1)] = empirical_FC
    empirical_FC_matrix = empirical_FC_matrix + empirical_FC_matrix.T
    # plot the empirical functional connectivity matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(empirical_FC_matrix, cmap='jet', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('Empirical FC matrix, ' + edge_weight + ', batch ' + str(batch_num))
    plt.savefig(predict_path + 'plots/' + 'empirical_FC_matrix_' + edge_weight + '_batch_' + str(batch_num) + '.png')
    plt.close()


def aicc_gls(gls_results):
    """
    Calculate the small sample size corrected Akaike Information Criterion (AICc)
    for Generalized Least Squares (GLS) results.

    Parameters:
        gls_results : statsmodels.regression.linear_model.RegressionResultsWrapper
            The GLS results obtained from statsmodels GLS.

    Returns:
        float
            The AICc value.
    """
    n = gls_results.nobs  # Number of observations
    k = len(gls_results.params)  # Number of model parameters
    aic = gls_results.aic  # Regular AIC from GLS results

    # Calculate AICc
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)

    return aicc



# get path to sf_prediction folder from the command line
predict_path = sys.argv[1]
print(predict_path)


# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

atlas = 'aal116'

# load clinical outcomes data
behavioral = pd.read_csv(
    str(predict_path) + '/clinical_prediction/group_data.csv')
print(behavioral.describe())

for i in [11]: # change this number for different batches of train-test splits
    for edge_weight in ['msmt_sift_invnodevol_radius2_count']: # change for different reconstruction + SC edge weights
        # get the predicted functional connectivity matrices and file_names from h5py as a dataframe
        pred_FC_with_labels = get_predicted_FC_with_labels(
            predict_path, edge_weight, i)
        print(pred_FC_with_labels.shape)
        print(pred_FC_with_labels.head())
        # get the empirical functional connectivity matrices and file_names from h5py as a dataframe
        emp_FC_with_labels = get_empirical_FC_with_labels(
            predict_path, edge_weight, i)
        print(emp_FC_with_labels.shape)
        # rename any integer columns to string columns with empirical_ prefix
        emp_FC_with_labels = emp_FC_with_labels.rename(
            columns=lambda x: 'empirical_' + str(x) if isinstance(x, int) else x)
        print(emp_FC_with_labels.head())
        # combine the predicted and empirical functional connectivity matrices dataframes
        data = pd.concat([pred_FC_with_labels, emp_FC_with_labels], axis=1)
        print(data.shape)
        print(data.head())
        data.to_csv(str(predict_path) + '/clinical_prediction/' + edge_weight + '_batch' + str(i) + '_data.csv')
        data = data.rename(columns={'file_name': 'subject_id'})
        # change behavioral Participant ID columns to subject_id
        behavioral = behavioral.rename(
            columns={'Participant_ID': 'subject_id'})
        # add sub-SAY to the front of the subject_id values
        behavioral['subject_id'] = 'sub-SAY' + \
            behavioral['subject_id'].astype(str)
        # change value for 'Sex' in behavioral dataframe to 0 if male 1 if female
        for ii in range(behavioral.shape[0]):
            if behavioral['Sex'][ii] == 'Male':
                behavioral['Sex'][ii] = 0
            elif behavioral['Sex'][ii] == 'Female':
                behavioral['Sex'][ii] = 1
        print(behavioral.head())
        # drop rows with NaN values in the behavioral dataframe
        behavioral = behavioral.dropna()
        # drop trial 2 of the 4 square step test
        behavioral = behavioral.drop(columns=['Balance 4 Square Step Test Time Trial 2'])
        # print dataframe describe to LaTeX file using tabulate
        desc_behavioral_df = behavioral.describe()
        latex_desc_behavioral_df = tabulate(desc_behavioral_df, tablefmt="latex", floatfmt=".2f", headers="keys")
        filename_desc = predict_path + '/results/' + edge_weight + '_behavioral_describe_batch_' + str(i) + '.tex'
        # print dataframe to LaTeX file using tabulate
        with open(filename_desc, "w+") as file:
            file.write(latex_desc_behavioral_df)
        # merge the behavioral data with the FC data
        data = pd.merge(data, behavioral, on='subject_id')
        data['Sex'] = pd.to_numeric(data['Sex'])
        # describe the data
        print(data.head())
        # perform first level of generalized least squares linear regression with the behavioral data as the outcome and the demographic data as the predictors
        # set predictors as the demographic data (columns Age and Sex and BMI) from the data dataframe
        predictors = data[['Age', 'Sex', 'BMI']]
        # predictors = data['Age']
        predictors = sm.add_constant(predictors)
        features = ["Peak VO2", "Balance 4 Square Step Test Time Trial 1", "Stair Climb Downstairs Time", "Stair Climb Upstairs Time"] 
        X = data[features].values
        # Perform factor analysis
        n_factors = 1  # Number of factors to extract
        fa = FactorAnalysis(n_components=n_factors, random_state=0)
        scores = fa.fit_transform(X)
        # Create a new column "Functional Fitness Score" in the DataFrame
        data["Functional Fitness Score"] = scores
        # make a copy of data df without the FC columns (i.e. 'empirical' or 'predicted')
        data_no_FC = data.drop(columns=['empirical_' + str(iii) for iii in range(6670)] + ['predicted_' + str(iii) for iii in range(6670)], inplace=False)
        # save data to a csv file
        data_no_FC.to_csv(predict_path + '/results/' + edge_weight + '_data_batch' + str(i) + '_data.csv')
        outcome = data["Functional Fitness Score"]
        # perform generalized least squares linear regression
        model = sm.GLS(outcome, predictors)
        results = model.fit()
        demographic_results = results
        # print the results
        print(results.summary())
        # save results to csv with date in filename
        filename = predict_path + '/results/' + edge_weight + 'GLS_AGE_batch_' + str(i) + '.csv'
        summary_table = results.summary().tables[1]
        with open(filename, "w+") as file:
            for line in summary_table:
                row = ""
                for word in line:
                    row = row + str(word) + ","
                row = row.rstrip(",")
                file.write(row + "\n")
        # save summary to latex table
        filename = predict_path + '/results/' + edge_weight + '_summary_table_GLS_AGE_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(summary_table.as_latex_tabular())
        # save summary to latex
        filename = predict_path + '/results/' + edge_weight + '_summary_GLS_AGE_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(results.summary().as_latex())
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig(predict_path + '/results/' + edge_weight + 'GLS_AGE_batch_' + str(i) + '.png')
        # perform PCA on the empirical FC data, predicted FC data
        # get the empirical FC data as an array from the data dataframe
        #     the empirical FC data is 6670 columns of the data dataframe that start with 'empirical_'
        FC_empirical_list = []
        for column in data.columns:
            if column.startswith('empirical_'):
                FC_empirical_list.append(column)
        FC_empirical_array = np.array(data[FC_empirical_list])
        # get the predicted FC data as an array from the data dataframe
        #     the predicted FC data is 6670 columns of the data dataframe that start with 'predicted_'
        FC_predicted_list = []
        for column in data.columns:
            if column.startswith('predicted_'):
                FC_predicted_list.append(column)
        FC_predicted_array = np.array(data[FC_predicted_list])
        # FC_empirical_array = np.array(data['emp_FC'].tolist())
        # FC_predicted_array = np.array(data['pred_FC'].tolist())
        FC_empirical_principal_components, FC_predicted_principal_components, FC_predicted_principal_components_independent = get_principal_components(
            FC_empirical_array, FC_predicted_array)
        # add the principal components to the dataframe as columns
        data = pd.concat([data, pd.DataFrame(FC_empirical_principal_components, columns=['FC_principal_components_' + str(iii)
                                                                                            for iii in range(FC_empirical_principal_components.shape[1])])], axis=1)
        data = pd.concat([data, pd.DataFrame(FC_predicted_principal_components, columns=['pred_FC_principal_components_' + str(iii)
                                                                                            for iii in range(FC_predicted_principal_components.shape[1])])], axis=1)
        data = pd.concat([data, pd.DataFrame(FC_predicted_principal_components_independent, columns=['pred_FC_principal_components_independent_' + str(iii)
                                                                                            for iii in range(FC_predicted_principal_components_independent.shape[1])])], axis=1)
        # perform first level of generalized least squares linear regression with the behavioral data as the outcome
        #     and the principal components of the empirical and predicted FC data as the predictors
        predictors_columns = ['Age', 'Sex', 'BMI']
        for iii in range(50):
            predictors_columns.append('FC_principal_components_' + str(iii))
        predictors = data[predictors_columns]
        predictors = sm.add_constant(predictors)
        # perform generalized least squares linear regression
        model = sm.GLS(outcome, predictors)
        results = model.fit()
        FC_empirical_results = results
        # print the results
        print(results.summary())
        # save results to csv
        summary_table_FC_empirical = results.summary().tables[1]
        with open(predict_path + '/results/' + edge_weight + 'GLS_FC_empirical_batch_' + str(i) + '.csv', "w+") as file:
            for line in summary_table_FC_empirical:
                row = ""
                for word in line:
                    row = row + str(word) + ","
                row = row.rstrip(",")
                file.write(row + "\n")
        # save summary to latex table
        filename = predict_path + '/results/' + edge_weight + '_summary_table_GLS_FC_empirical_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(summary_table_FC_empirical.as_latex_tabular())
        # save summary to latex
        filename = predict_path + '/results/' + edge_weight + '_summary_GLS_FC_empirical_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(results.summary().as_latex())
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig(predict_path + '/results/' + edge_weight + 'GLS_FC_empirical_batch_' + str(i) + '.png')

        # set predictors as the first fifty principal components of the predicted FC data
        predictors_columns = ['Age', 'Sex', 'BMI']
        for iii in range(50):
            predictors_columns.append('pred_FC_principal_components_' + str(iii))
        predictors = data[predictors_columns]
        predictors = sm.add_constant(predictors)
        model = sm.GLS(outcome, predictors)
        results = model.fit()
        FC_predicted_results = results
        # print the results
        print(results.summary())
        # save results to csv
        summary_table_FC_predicted = results.summary().tables[1]
        with open(predict_path + '/results/' + edge_weight + 'GLS_FC_predicted_batch_' + str(i) + '.csv', "w+") as file:
            for line in summary_table_FC_predicted:
                row = ""
                for word in line:
                    row = row + str(word) + ","
                row = row.rstrip(",")
                file.write(row + "\n")
        # save summary to latex table
        filename = predict_path + '/results/' + edge_weight + '_summary_table_GLS_FC_predicted_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(summary_table_FC_predicted.as_latex_tabular())
        # save summary to latex
        filename = predict_path + '/results/' + edge_weight + '_summary_GLS_FC_predicted_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(results.summary().as_latex())
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig(predict_path + '/results/' + edge_weight + 'GLS_FC_predicted_batch_' + str(i) + '.png')
        # set predictors as the first fifty principal components of the predicted FC data
        predictors_columns = ['Age', 'Sex', 'BMI']
        for iii in range(50):
            predictors_columns.append('pred_FC_principal_components_independent_' + str(iii))
        predictors = data[predictors_columns]
        predictors = sm.add_constant(predictors)
        model = sm.GLS(outcome, predictors)
        results = model.fit()
        FC_predicted_independent_results = results
        # print the results
        print(results.summary())
        # save results to csv
        summary_table_FC_predicted_independent = results.summary().tables[1]
        with open(predict_path + '/results/' + edge_weight + 'GLS_FC_predicted_independent_batch_' + str(i) + '.csv', "w+") as file:
            for line in summary_table_FC_predicted_independent:
                row = ""
                for word in line:
                    row = row + str(word) + ","
                row = row.rstrip(",")
                file.write(row + "\n")
        # save summary to latex table
        filename = predict_path + '/results/' + edge_weight + '_summary_table_GLS_FC_predicted_independent_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(summary_table_FC_predicted_independent.as_latex_tabular())
        # save summary to latex
        filename = predict_path + '/results/' + edge_weight + '_summary_GLS_FC_predicted_independent_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(results.summary().as_latex())
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig(predict_path + '/results/' + edge_weight + 'GLS_FC_predicted_independent_batch_' + str(i) + '.png')
        # perform second level of generalized least squares linear regression with the behavioral data as the outcome
        #     and the principal components of the empirical and predicted FC data as the predictors
        # set predictors as the first fifty principal components of the empirical FC data, data['Age'], and data['Sex'], and BMI
        predictors_columns = ['Age', 'Sex', 'BMI']
        for iii in range(50):
            predictors_columns.append('FC_principal_components_' + str(iii))
        for iii in range(50):
            predictors_columns.append('pred_FC_principal_components_' + str(iii))
        predictors = data[predictors_columns]
        predictors = sm.add_constant(predictors)
        # perform generalized least squares linear regression
        model = sm.GLS(outcome, predictors)
        results = model.fit()
        FC_empirical_and_predicted_results = results
        # print the results
        print(results.summary())
        # save results to csv
        summary_table_FC_empirical = results.summary().tables[1]
        with open(predict_path + '/results/' + edge_weight + 'GLS_FC_empirical_and_predicted_batch_' + str(i) + '.csv', "w+") as file:
            for line in summary_table_FC_empirical:
                row = ""
                for word in line:
                    row = row + str(word) + ","
                row = row.rstrip(",")
                file.write(row + "\n")
        # save summary to latex table
        filename = predict_path + '/results/' + edge_weight + '_summary_table_GLS_FC_empirical_and_predicted_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(summary_table_FC_empirical.as_latex_tabular())
        # save summary to latex
        filename = predict_path + '/results/' + edge_weight + '_summary_GLS_FC_empirical_and_predicted_batch_' + str(i) + '.tex'
        with open(filename, "w+") as file:
            file.write(results.summary().as_latex())
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig(predict_path + '/results/' + edge_weight + 'GLS_FC_empirical_and_predicted_batch_' + str(i) + '.png')
        # calculate AICc with aicc_gls function
        demographic_results_aicc = aicc_gls(demographic_results)
        FC_empirical_results_aicc = aicc_gls(FC_empirical_results)
        FC_predicted_results_aicc = aicc_gls(FC_predicted_results)
        FC_predicted_independent_results_aicc = aicc_gls(FC_predicted_independent_results)
        FC_empirical_and_predicted_results_aicc = aicc_gls(FC_empirical_and_predicted_results)
        # compare akaike information criterion (AIC) and adjusted for small sample sizes (AICc) for the four models (demographic, empirical, predicted, predicted independent, empirical and predicted)
        print('AIC for demographic model: ' + str(demographic_results.aic) + ' AICc: ' + str(demographic_results_aicc))
        print('AIC for empirical model: ' + str(FC_empirical_results.aic) + ' AICc: ' + str(FC_empirical_results_aicc))
        print('AIC for predicted model: ' + str(FC_predicted_results.aic) + ' AICc: ' + str(FC_predicted_results_aicc))
        print('AIC for predicted independent model: ' + str(FC_predicted_independent_results.aic) + ' AICc: ' + str(FC_predicted_independent_results_aicc))
        print('AIC for empirical and predicted model: ' + str(FC_empirical_and_predicted_results.aic) + ' AICc: ' + str(FC_empirical_and_predicted_results_aicc))
        # create a dataframe with the AIC values
        AIC_df = pd.DataFrame({'model': ['demographic', 'empirical', 'predicted', 'predicted independent', 'empirical and predicted'], 'AIC': [demographic_results.aic, FC_empirical_results.aic, FC_predicted_results.aic, FC_predicted_independent_results.aic, FC_empirical_and_predicted_results.aic], 'AICc': [demographic_results_aicc, FC_empirical_results_aicc, FC_predicted_results_aicc, FC_predicted_independent_results_aicc, FC_empirical_and_predicted_results_aicc]})
        # drop the rows for 'demographic' and 'empirical and predicted'
        AIC_df = AIC_df.drop([0, 4])
        # compare probability to minimize information loss based on AIC
        AIC_df['delta_AIC'] = AIC_df['AIC'] - np.min(AIC_df['AIC'])
        AIC_df['probability_AIC'] = np.exp(-0.5 * AIC_df['delta_AIC'])
        # compare probability to minimize information loss based on AICc
        AIC_df['delta_AICc'] = AIC_df['AICc'] - np.min(AIC_df['AICc'])
        AIC_df['probability_AICc'] = np.exp(-0.5 * AIC_df['delta_AICc'])
        # save AIC_df to csv
        AIC_df.to_csv(predict_path + '/results/' + edge_weight + '_AICc_df_' + str(i) + '.csv')
        # plot the AIC values
        plt.figure()
        plt.bar(AIC_df['model'], AIC_df['AIC'])
        plt.ylabel('AIC')
        plt.savefig(predict_path + '/results/' + edge_weight + '_AIC_' + str(i) + '.png')
        # plot the AICc values
        plt.figure()
        plt.bar(AIC_df['model'], AIC_df['AICc'])
        plt.ylabel('AICc')
        plt.savefig(predict_path + '/results/' + edge_weight + '_AICc_' + str(i) + '.png')
        # compare the p-values of the four models
        print('p-value for demographic model: ' + str(demographic_results.f_pvalue))
        print('p-value for empirical model: ' + str(FC_empirical_results.f_pvalue))
        print('p-value for predicted model: ' + str(FC_predicted_results.f_pvalue))
        print('p-value for predicted independent model: ' + str(FC_predicted_independent_results.f_pvalue))
        print('p-value for empirical and predicted model: ' + str(FC_empirical_and_predicted_results.f_pvalue))
        # create a dataframe with the p-values
        p_df = pd.DataFrame({'model': ['demographic', 'empirical', 'predicted', 'predicted independent', 'empirical and predicted'], 'p-value': [demographic_results.f_pvalue, FC_empirical_results.f_pvalue, FC_predicted_results.f_pvalue, FC_predicted_independent_results.f_pvalue, FC_empirical_and_predicted_results.f_pvalue]})
        # save p_df to csv
        p_df.to_csv(predict_path + '/results/' + edge_weight + '_p_df_' + str(i) + '.csv')
