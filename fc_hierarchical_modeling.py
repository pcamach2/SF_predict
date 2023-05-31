# import packages
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import pymc3 as pm 
import arviz as az
import pickle

# %config InlineBackend.figure_format = 'retina'
az.style.use("arviz-darkgrid")

atlas = 'aal116'

# # load clinical outcomes data
# behavioral = pd.read_csv('/home/paul/thesis/dev/SAY_sf_prediction_v2/group_data.csv')

# load data for each dataset (1-100) of each edge weight (gqi_count_sum, msmt_sift_radius2_count, dti_dti_count)
for i in [90, 91]: # np.arange(100):
    for edge_weight in  ['dti_dti_volumeweighted']:  # 'gqi_count_sum', 'msmt_sift_radius2_count', 'dti_dti_count':
        with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/preds_percent_' + atlas + '_' + edge_weight + '_diag1_n' + str(i) + '.pkl', 'rb') as f_pred:
            preds = pickle.load(f_pred)
            pred_FC = preds[4]['test'] # this needs to be adapted based on how the pickle loads the dictionary of lists, which walk lengths are included, etc.
        # we should pull the train data too!
        # perform hierarchical regression with pymc3 to predict clinical outcomes from first predicted functional connectivity, 
        #     then empirical functional connectivity
        # load empirical functional connectivity data from h5py
        fc_test = h5py.File('/home/paul/thesis/dev/SAY_sf_prediction_v2/dataset/test_percent_' + edge_weight + '_diag1_' + str(i) + '.h5py', 'r')
        fc_test = list(np.array(fc_test.get('labels')))
        # set up empty array to store flattened upper triangle of fc_test matrices for each subject
        FC_test = np.zeros((len(fc_test), 6670)) # 6670 is the number of edges in the upper triangle of a 116x116 matrix
        # get flattened upper triangle of fc_test matrices for each subject
        for i in range(len(fc_test)):
            FC_test[i] = fc_test[i][np.triu_indices(116, k = 1)]

        # Simulate clinical outcome variable
        clinical_outcome = np.random.rand(21)
        
        # Create a design matrix with predictors
        predictors = np.concatenate((pred_FC, FC_test), axis=1)

        # Define the hierarchical model for linear regression
        with pm.Model() as hierarchical_model:
            # Priors for the model parameters
            alpha = pm.Normal("alpha", mu=0, sd=1) # , shape=predictors.shape[1])
            beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])
            # Linear regression equation
            mu = alpha + pm.math.dot(predictors, beta)
            # Likelihood function
            sigma = pm.HalfNormal("sigma", sd=1)
            clinical_outcome_obs = pm.Normal("clinical_outcome_obs", mu=mu, sd=sigma, observed=clinical_outcome)
            # Run the MCMC sampling
            trace = pm.sample(2000, tune=1000)
        # Add statistics to a pandas dataframe
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
        # # Plotting posterior distributions
        # for ii in range(predictors.shape[1]):
        #     pm.plot_posterior(trace["beta"][:, ii], credible_interval=0.95)
        #     plt.xlabel("Beta")
        #     plt.ylabel("Density")
        #     plt.title(f"Posterior Distribution of Beta {ii+1}")
        #     # save figure with edge_weight and dataset number in the filename
        #     plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_posterior_beta_predictor_' + str(ii) + '_' + edge_weight + '_dataset_n' + str(i) + '.png')
        #     plt.show()
        # Plotting trace plots
        pm.traceplot(trace)
        # save figure with edge_weight and dataset number in the filename
        plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.png')
        plt.show()
        plt.close()
        # Plotting posterior predictive checks
        with hierarchical_model:
            ppc = pm.sample_posterior_predictive(trace, samples=1000)
        az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model))
        plt.xlabel("Clinical Outcome")
        plt.ylabel("Density")
        plt.title("Posterior Predictive Distribution")
        # save figure with edge_weight and dataset number in the filename
        plt.savefig('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.png')
        plt.show()
        plt.close()
        #
        # save the trace and posterior predictive checks as pickle files
        with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_trace:
            pickle.dump(trace, f_trace)
        with open('/home/paul/thesis/dev/SAY_sf_prediction_v2/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_ppc:
            pickle.dump(ppc, f_ppc)

        # # run hierarchical bayesian regression with pymc3 to predict clinical outcomes from first predicted functional connectivity,
        # #     then empirical functional connectivity, then demographic variables
        # #
        # ## Simulate clinical outcome variable
        # # clinical_outcome = np.random.rand(n_samples)
        # #
        # # Create a design matrix with predictors
        # predictors = np.concatenate((pred_FC, fc_test, behavioral), axis=1)
        # # Define the hierarchical model for bayesian regression
        # with pm.Model() as hierarchical_model:
        #     # Priors for the model parameters
        #     alpha = pm.Normal("alpha", mu=0, sd=1)
        #     beta = pm.Normal("beta", mu=0, sd=1, shape=predictors.shape[1])
        #     # Linear regression equation
        #     mu = alpha + pm.math.dot(predictors, beta)
        #     # Likelihood function
        #     sigma = pm.HalfNormal("sigma", sd=1)
        #     clinical_outcome_obs = pm.Normal("clinical_outcome_obs", mu=mu, sd=sigma, observed=clinical_outcome)
        #     # Run the MCMC sampling
        #     trace = pm.sample(2000, tune=1000)
        # # Add statistics to a pandas dataframe
        # df_summary = pm.summary(trace)
        # # Save the dataframe as a csv file with edge_weight and dataset number in the filename
        # df_summary.to_csv('/dataout/hierarchical_model_summary_' + edge_weight + '_n' + str(i) + '.csv')
        # # Print the summary of the posterior distribution
        # pm.summary(trace)
        # # Plotting posterior distributions
        # pm.plot_posterior(trace["alpha"], credible_interval=0.95)
        # plt.xlabel("Alpha")
        # plt.ylabel("Density")
        # plt.title("Posterior Distribution of Alpha")
        # # save figure with edge_weight and dataset number in the filename
        # plt.savefig('/dataout/hierarchical_model_posterior_alpha_' + edge_weight + '_dataset_n' + str(i) + '.png')
        # plt.show()
        # plt.close()
        # # Plotting posterior distributions
        # for ii in range(predictors.shape[1]):
        #     pm.plot_posterior(trace["beta"][:, ii], credible_interval=0.95)
        #     plt.xlabel("Beta")
        #     plt.ylabel("Density")
        #     plt.title(f"Posterior Distribution of Beta {ii+1}")
        #     # save figure with edge_weight and dataset number in the filename
        #     plt.savefig('/dataout/hierarchical_model_posterior_beta_predictor_' + str(ii) + '_' + edge_weight + '_dataset_n' + str(i) + '.png')
        #     plt.show()
        # # Plotting trace plots
        # pm.traceplot(trace)
        # # save figure with edge_weight and dataset number in the filename
        # plt.savefig('/dataout/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.png')
        # plt.show()
        # plt.close()
        # # Plotting posterior predictive checks
        # with hierarchical_model:
        #     ppc = pm.sample_posterior_predictive(trace, samples=1000)
        # az.plot_ppc(az.from_pymc3(posterior_predictive=ppc, model=hierarchical_model))
        # plt.xlabel("Clinical Outcome")
        # plt.ylabel("Density")
        # plt.title("Posterior Predictive Distribution")
        # # save figure with edge_weight and dataset number in the filename
        # plt.savefig('/dataout/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.png')
        # plt.show()
        # plt.close()
        # #
        # # save the trace and posterior predictive checks as pickle files
        # with open('/dataout/hierarchical_model_trace_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_trace:
        #     pickle.dump(trace, f_trace)
        # with open('/dataout/hierarchical_model_ppc_' + edge_weight + '_dataset_n' + str(i) + '.pkl', 'wb') as f_ppc:
        #     pickle.dump(ppc, f_ppc)







