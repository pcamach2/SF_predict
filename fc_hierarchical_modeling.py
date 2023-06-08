# import packages
import os
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
import statsmodels.api as sm

def get_indices_first_ten_factor_analysis(X):
    # Perform Factor Analysis
    fa = FactorAnalyzer(n_factors=10)
    fa.fit(X)

    # Get the indices of the first ten factors
    indices = np.argwhere(np.sum(fa.loadings_[:, :10], axis=1) != 0).flatten()

    # Create a DataFrame with factor loadings
    factors_df = pd.DataFrame(fa.loadings_[:, :fa.n_factors_], columns=['Factor {}'.format(i+1) for i in range(fa.n_factors_)])

    # Save DataFrame to a CSV file
    factors_df.to_csv('factor_analysis_loadings.csv', index=False)

    # Plot factor loadings
    plt.figure(figsize=(10, 6))
    sns.barplot(data=factors_df)
    plt.title('Factor Loadings')
    plt.xlabel('Features')
    plt.ylabel('Loadings')
    plt.xticks(rotation=90)
    plt.show()

    return indices

def get_indices_first_n_pca(X, n_components=10):
    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Get the indices of the first ten components
    indices = np.argwhere(np.sum(pca.components_[:n_components], axis=0) != 0).flatten()

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
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test

def get_principal_components(original_matrix, new_matrix, num_components):
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
    original_principal_components = np.dot(original_matrix_scaled, weightings.T)

    # Principal components of new matrix
    new_principal_components = np.dot(new_matrix_scaled, weightings.T)

    return original_principal_components, new_principal_components

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

atlas = 'aal116'

# # # load clinical outcomes data
# behavioral = pd.read_csv('/home/paul/thesis/dev/SAY_sf_prediction_v2/group_data.csv')
# # summarize behavioral data
# print(behavioral.describe())

# load demographic data
demographic = pd.read_csv('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/demographic.csv')
# summarize demographic data
print(demographic.describe())


# load data for each dataset (1-100) of each edge weight (gqi_count_sum, msmt_sift_radius2_count, dti_dti_count)
for i in [25,26]: # np.arange(100):
    for edge_weight in  ['gqi_count_sum']:  # 'gqi_count_sum', 'msmt_sift_radius2_count', 'dti_dti_count':
        with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/preds_percent_' + atlas + '_' + edge_weight + '_diag1_n' + str(i) + '.pkl', 'rb') as f_pred:
            preds = pickle.load(f_pred)
            pred_FC = preds[4]['test'] # this needs to be adapted based on how the pickle loads the dictionary of lists, which walk lengths are included, etc.
        # we should pull the train data too!
        # perform hierarchical regression with pymc3 to predict clinical outcomes from first predicted functional connectivity, 
        #     then empirical functional connectivity
        # load empirical functional connectivity data from h5py
        fc_test = h5py.File('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/test_percent_' + edge_weight + '_diag1_' + str(i) + '.h5py', 'r')
        fc_test = list(np.array(fc_test.get('labels')))
        # get the file_names for each subject
        file_names_fc_test = h5py.File('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/test_percent_' + edge_weight + '_diag1_' + str(i) + '.h5py', 'r')
        file_names_fc_test = list(np.array(file_names_fc_test.get('file_name')))
        # set up empty array to store flattened upper triangle of fc_test matrices for each subject
        FC_test = np.zeros((len(fc_test), 6670)) # 6670 is the number of edges in the upper triangle of a 116x116 matrix
        # get flattened upper triangle of fc_test matrices for each subject
        for ii in range(len(fc_test)):
            FC_test[ii] = fc_test[ii][np.triu_indices(116, k = 1)]
        # Perform PCA and get the first ten components of the empirical functional connectivity data 
        #     and predicted functional connectivity data for each subject
        FC_test_principal_components, pred_test_FC_principal_components = get_principal_components(FC_test, pred_FC, 10)

        # perform the above for the training data too!
        # load empirical functional connectivity data from h5py
        fc_train = h5py.File('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/train_percent_' + edge_weight + '_diag1_' + str(i) + '.h5py', 'r')
        fc_train = list(np.array(fc_train.get('labels')))
        # get the file_names for each subject
        file_names_fc_train = h5py.File('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/train_percent_' + edge_weight + '_diag1_' + str(i) + '.h5py', 'r')
        file_names_fc_train = list(np.array(file_names_fc_train.get('file_name')))
        # set up empty array to store flattened upper triangle of fc_test matrices for each subject
        FC_train = np.zeros((len(fc_train), 6670)) # 6670 is the number of edges in the upper triangle of a 116x116 matrix
        # get flattened upper triangle of fc_test matrices for each subject
        for ii in range(len(fc_train)):
            FC_train[ii] = fc_train[ii][np.triu_indices(116, k = 1)]
        # get teh predicted FC for the training data
        pred_FC_train = preds[4]['train']
        # Perform PCA and get the first ten components of the empirical functional connectivity data
        #     and predicted functional connectivity data for each subject
        FC_train_principal_components, pred_train_FC_principal_components = get_principal_components(FC_train, pred_FC_train, 10)
        # create a dataframe to combine the principal components of the empirical FC and predicted FC data with
        #      the matched behavioral data based on the file_name value from the respective h5py files for training and testing
        #      and the subject_id value from the behavioral data
        # get the file_name values from the h5py files
        file_names_test = []
        for iii in range(len(fc_test)):
            file_names_test.append(file_names_fc_test[iii])
        file_names_train = []
        for iii in range(len(fc_train)):
            file_names_train.append(file_names_fc_train[iii])
        # concatenate the file_name values from the h5py files
        file_names_dev = file_names_test + file_names_train
        # since we have no behavioral data yet, simulate the behavioral dataframe with random values and the same subject ids as the h5py files
        behavioral = pd.DataFrame({'subject_id': file_names_dev})
        # populate the behavioral dataframe with random values for the clinical outcomes
        behavioral['clinical_outcome'] = np.random.randint(0, 2, behavioral.shape[0])

        # get the subject_id values from the behavioral data
        subject_ids = behavioral['subject_id']
        # create a dataframe for the test data with the first ten principal components of the empirical FC, such that each weight in the principal components is a separate column
        test_data = pd.DataFrame(FC_test_principal_components, columns = ['FC_principal_components_' + str(iii) for iii in range(10)])        
        # create a dataframe for the test data with the first ten principal components of the predicted FC, such that each weight in the principal components is a separate column
        test_data = pd.concat([test_data, pd.DataFrame(pred_test_FC_principal_components, columns = ['pred_FC_principal_components_' + str(iii) for iii in range(10)])], axis = 1)
        # add the file_name values from the h5py files to the test data
        test_data['file_name'] = file_names_test
        # create a dataframe for the training data with the first ten principal components of the empirical FC, such that each weight in the principal components is a separate column
        train_data = pd.DataFrame(FC_train_principal_components, columns = ['FC_principal_components_' + str(iii) for iii in range(10)])
        # create a dataframe for the training data with the first ten principal components of the predicted FC, such that each weight in the principal components is a separate column
        train_data = pd.concat([train_data, pd.DataFrame(pred_train_FC_principal_components, columns = ['pred_FC_principal_components_' + str(iii) for iii in range(10)])], axis = 1)
        # add the file_name values from the h5py files to the training data
        train_data['file_name'] = file_names_train
        # merge the training and testing dataframes
        data = pd.concat([train_data, test_data], axis=0)
        # change the file_name column to subject_id
        data = data.rename(columns={'file_name': 'subject_id'})
        # add 'sub-' to the front of the subject_id values
        data['subject_id'] = 'sub-' + data['subject_id'].astype(str)
        # describe the data
        data.describe()
        # merge the behavioral data with the FC data
        data = pd.merge(data, behavioral, on='subject_id')
        # describe the data
        data.describe()
        # # drop the subject_id and file_name columns
        # data = data.drop(['subject_id', 'file_name'], axis=1)
        # # drop the rows with NaN values
        # data = data.dropna()
        # # reset the index
        # data = data.reset_index(drop=True)

        # perform first level of ordinary least squares linear regression with the behavioral data as the outcome and the demographic data as the predictors
        # set predictors as the demographic data (Age, Sex)
        predictors = demographic.drop(['subject_id', 'file_name'], axis=1)
        # set outcome as the clinical outcome
        outcome = data['clinical_outcome']
        # perform ordinary least squares linear regression
        model = sm.OLS(outcome, predictors)
        results = model.fit()
        # print the results
        print(results.summary())
        # save results to csv
        results.summary().tables[1].as_csv('results/OLS_demographic.csv')
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig('results/OLS_demographic.png')
        
        # perform first level of ordinary least squares linear regression with the behavioral data as the outcome and the principal components of the
        #     empirical and predicted FC data as the predictors
        # set predictors as the first ten principal components of the empirical FC data
        predictors = data[['FC_principal_components_' + str(iii) for iii in range(10)]]
        # set outcome as the clinical outcome
        outcome = data['clinical_outcome']
        # perform ordinary least squares linear regression
        model = sm.OLS(outcome, predictors)
        results = model.fit()
        # print the results
        print(results.summary())
        # save results to csv
        results.summary().tables[1].as_csv('results/OLS_FC_empirical.csv')
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig('results/OLS_FC.png')

        # set predictors as the first ten principal components of the predicted FC data
        predictors = data[['pred_FC_principal_components_' + str(iii) for iii in range(10)]]
        # set outcome as the clinical outcome
        outcome = data['clinical_outcome']
        # perform ordinary least squares linear regression
        model = sm.OLS(outcome, predictors)
        results = model.fit()
        # print the results
        print(results.summary())
        # save results to csv
        results.summary().tables[1].as_csv('results/OLS_FC.csv')
        # create and save a plot of the results
        plt.figure()
        plt.plot(results.fittedvalues, outcome, 'o')
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.savefig('results/OLS_FC_predicted.png')
        

        # perform hierarchical linear regression with the behavioral data as the outcome and the demographic data and
        #     principal components of the empirical and predicted FC data as the predictors
        # Create a design matrix with predictors for the hierarchical model for linear regression
        #     using each of the first ten principal components of the empirical and predicted FC data
        predictors = np.zeros((data.shape[0], 20))
        predictors[:, :10] = data[['FC_principal_components_' + str(iii) for iii in range(10)]]
        predictors[:, 10:] = data[['pred_FC_principal_components_' + str(iii) for iii in range(10)]]
        # Define the hierarchical model for linear regression
        with pm.Model() as hierarchical_model:
            # Priors for the model parameters
            alpha = pm.Normal("alpha", mu=0, sd=1) # , shape=predictors.shape[1])
            beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])
            # Linear regression equation
            mu = alpha + pm.math.dot(predictors, beta)
            # Likelihood of observations
            y = pm.Normal("y", mu=mu, sd=1, observed=data['clinical_outcome'])
            # Sample posterior distribution
            trace = pm.sample(1000, tune=1000, cores=1)
        # Get the posterior samples
        posterior_samples = pm.trace_to_dataframe(trace)
        # # Get the posterior predictive samples
        # posterior_predictive_samples = pm.sample_posterior_predictive(trace, samples=1000, model=hierarchical_model)
        # # Get the posterior predictive mean
        # posterior_predictive_mean = posterior_predictive_samples['y'].mean(axis=0)
        # # Get the posterior predictive standard deviation
        # posterior_predictive_std = posterior_predictive_samples['y'].std(axis=0)
        # # Get the posterior predictive 95% confidence interval
        # posterior_predictive_95CI = np.percentile(posterior_predictive_samples['y'], [2.5, 97.5], axis=0)
        # # # Get the posterior predictive 50% confidence interval
        # # posterior_predictive_50CI = np.percentile(posterior_predictive_samples['y'], [25, 75], axis=0)
        # # # Get the posterior predictive 25% confidence interval
        # # posterior_predictive_25CI = np.percentile(posterior_predictive_samples['y'], [12.5, 87.5], axis=0)
        # # # Get the posterior predictive 5% confidence interval
        # # posterior_predictive_5CI = np.percentile(posterior_predictive_samples['y'], [2.5, 97.5], axis=0)
        # # # Get the posterior predictive 2.5% confidence interval
        # # posterior_predictive_2p5CI = np.percentile(posterior_predictive_samples['y'], [1.25, 98.75], axis=0)
        # # Add statistics to a pandas dataframe
        df_summary = pm.summary(trace)
        # Save the dataframe as a csv file with edge_weight and dataset number in the filename
        df_summary.to_csv('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_summary_' + edge_weight + '_n' + str(i) + '.csv')
        # Print the summary of the posterior distribution
        pm.summary(trace)
        # Plotting posterior distributions
        pm.plot_posterior(trace["alpha"]) #, credible_interval=0.95) # this threw an error for unexpected argument credible_interval
        plt.xlabel("Alpha")
        plt.ylabel("Density")
        plt.title("Posterior Distribution of Alpha")
        # save figure with edge_weight and dataset number in the filename
        plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_posterior_alpha_' + edge_weight + '_dataset_n' + str(i) + '.png')
        plt.show()
        plt.close()




          # train_behavioral, test_behavioral = split_data(behavioral, test_size=0.2, random_state=i)
          #
          # combine the FC data with the behavioral data
            # train_behavioral = train_behavioral.reset_index(drop=True)
            # test_behavioral = test_behavioral.reset_index(drop=True)


#         # # Create a design matrix with predictors
#         # predictors = np.concatenate((pred_FC, FC_test), axis=1)

#         # # Define the hierarchical model for linear regression
#         # with pm.Model() as hierarchical_model:
#         #     # Priors for the model parameters
#         #     alpha = pm.Normal("alpha", mu=0, sd=1) # , shape=predictors.shape[1])
#         #     beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])
#         #     # Linear regression equation
#         #     mu = alpha + pm.math.dot(predictors, beta)
#         #     # Likelihood function
#         #     sigma = pm.HalfNormal("sigma", sd=1)
#         #     clinical_outcome_obs = pm.Normal("clinical_outcome_obs", mu=mu, sd=sigma, observed=clinical_outcome)
#         #     # Run the MCMC sampling
#         #     trace = pm.sample(2000, tune=1000)
#         # # Add statistics to a pandas dataframe
#         # df_summary = pm.summary(trace)
#         # # Save the dataframe as a csv file with edge_weight and dataset number in the filename
#         # df_summary.to_csv('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_summary_' + edge_weight + '_n' + str(i) + '.csv')
#         # # Print the summary of the posterior distribution
#         # pm.summary(trace)
#         # # Plotting posterior distributions
#         # pm.plot_posterior(trace["alpha"]) #, credible_interval=0.95) # this threw an error for unexpected argument credible_interval
#         # plt.xlabel("Alpha")
#         # plt.ylabel("Density")
#         # plt.title("Posterior Distribution of Alpha")
#         # # save figure with edge_weight and dataset number in the filename
#         # plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_posterior_alpha_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # plt.show()
#         # plt.close()
#         # # # Plotting posterior distributions
#         # # for ii in range(predictors.shape[1]):
#         # #     pm.plot_posterior(trace["beta"][:, ii], credible_interval=0.95)
#         # #     plt.xlabel("Beta")
#         # #     plt.ylabel("Density")
#         # #     plt.title(f"Posterior Distribution of Beta {ii+1}")
#         # #     # save figure with edge_weight and dataset number in the filename
#         # #     plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_posterior_beta_predictor_' + str(ii) + '_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # #     plt.show()
#         # # Plotting trace plots
#         # pm.traceplot(trace)
#         # # save figure with edge_weight and dataset number in the filename
#         # plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # plt.show()
#         # plt.close()
#         # # Plotting posterior predictive checks
#         # with hierarchical_model:
#         #     ppc = pm.sample_posterior_predictive(trace, samples=1000)
#         # az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model))
#         # plt.xlabel("Clinical Outcome")
#         # plt.ylabel("Density")
#         # plt.title("Posterior Predictive Distribution")
#         # # save figure with edge_weight and dataset number in the filename
#         # plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # plt.show()
#         # plt.close()
#         # #
#         # # save the trace and posterior predictive checks as pickle files
#         # with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_trace:
#         #     pickle.dump(trace, f_trace)
#         # with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_ppc:
#         #     pickle.dump(ppc, f_ppc)

#         # # # run hierarchical bayesian regression with pymc3 to predict clinical outcomes from first predicted functional connectivity,
#         # # #     then empirical functional connectivity, then demographic variables
#         # # #
#         # # ## Simulate clinical outcome variable
#         # # # clinical_outcome = np.random.rand(n_samples)
#         # # #
#         # # # Create a design matrix with predictors
#         # # predictors = np.concatenate((pred_FC, fc_test, behavioral), axis=1)
#         # # # Define the hierarchical model for bayesian regression
#         # # with pm.Model() as hierarchical_model:
#         # #     # Priors for the model parameters
#         # #     alpha = pm.Normal("alpha", mu=0, sd=1)
#         # #     beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])
#         # #     # Linear regression equation
#         # #     mu = alpha + pm.math.dot(predictors, beta)
#         # #     # Likelihood function
#         # #     sigma = pm.HalfNormal("sigma", sd=1)
#         # #     clinical_outcome_obs = pm.Normal("clinical_outcome_obs", mu=mu, sd=sigma, observed=clinical_outcome)
#         # #     # Run the MCMC sampling
#         # #     trace = pm.sample(2000, tune=1000)
#         # # # Add statistics to a pandas dataframe
#         # # df_summary = pm.summary(trace)
#         # # # Save the dataframe as a csv file with edge_weight and dataset number in the filename
#         # # df_summary.to_csv('/dataout/hierarchical_model_summary_' + edge_weight + '_n' + str(i) + '.csv')
#         # # # Print the summary of the posterior distribution
#         # # pm.summary(trace)
#         # # # Plotting posterior distributions
#         # # pm.plot_posterior(trace["alpha"], credible_interval=0.95)
#         # # plt.xlabel("Alpha")
#         # # plt.ylabel("Density")
#         # # plt.title("Posterior Distribution of Alpha")
#         # # # save figure with edge_weight and dataset number in the filename
#         # # plt.savefig('/dataout/hierarchical_model_posterior_alpha_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # # plt.show()
#         # # plt.close()
#         # # # Plotting posterior distributions
#         # # for ii in range(predictors.shape[1]):
#         # #     pm.plot_posterior(trace["beta"][:, ii], credible_interval=0.95)
#         # #     plt.xlabel("Beta")
#         # #     plt.ylabel("Density")
#         # #     plt.title(f"Posterior Distribution of Beta {ii+1}")
#         # #     # save figure with edge_weight and dataset number in the filename
#         # #     plt.savefig('/dataout/hierarchical_model_posterior_beta_predictor_' + str(ii) + '_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # #     plt.show()
#         # # # Plotting trace plots
#         # # pm.traceplot(trace)
#         # # # save figure with edge_weight and dataset number in the filename
#         # # plt.savefig('/dataout/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # # plt.show()
#         # # plt.close()
#         # # # Plotting posterior predictive checks
#         # # with hierarchical_model:
#         # #     ppc = pm.sample_posterior_predictive(trace, samples=1000)
#         # # az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model))
#         # # plt.xlabel("Clinical Outcome")
#         # # plt.ylabel("Density")
#         # # plt.title("Posterior Predictive Distribution")
#         # # # save figure with edge_weight and dataset number in the filename
#         # # plt.savefig('/dataout/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.png')
#         # # plt.show()
#         # # plt.close()
#         # # #
#         # # # save the trace and posterior predictive checks as pickle files
#         # # with open('/dataout/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_trace:
#         # #     pickle.dump(trace, f_trace)
#         # # with open('/dataout/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_ppc:
#         # #     pickle.dump(ppc, f_ppc)







