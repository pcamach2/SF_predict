import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import scipy.stats as stats
from scipy.stats import ttest_rel


# Read data
# main_path = '/home/paul/thesis/dev/SAY_sf_prediction_v3/dataset/'
# main_path = '/home/paul/thesis/dev/SAY_sf_prediction_v3/scrambled_dataset/'


main_path = '/datain/dataset'
file_list = Path(main_path).glob('*batch??.csv')
for file in file_list:
    print(file)
    df = pd.read_csv(file)
    # ,ave to a copy of df, then perform Fisher's r to z transformation on Pearson scores
    df_z = df.copy()
    for i in range(1, 6):
        df_z[str(i)] = np.arctanh(df_z[str(i)])
    # print(df_z)
    print(file.stem)
    # compare difference in score between Recon methods for each walk length column
    for i in range(1, 6):
        print('Walk Length: '+str(i))
        # get unique values of Recon column
        recon_list = df['Recon'].unique()
        df_stats_all = pd.DataFrame()
        df_stats = pd.DataFrame()
        # loop through each unique value of Recon column
        for recon in recon_list:
            print(recon)
            print(df[df['Recon'] == str(recon)][str(i)])
            print('Statistics for '+str(recon))
            print('Mean')
            print(df[df['Recon'] == str(recon)][str(i)].mean())
            print('Stdev')
            print(df[df['Recon'] == str(recon)][str(i)].std())
            print('Median')
            print(df[df['Recon'] == str(recon)][str(i)].median())
            print('IQR')
            print(df[df['Recon'] == str(recon)][str(i)].quantile(
                q=0.75)-df[df['Recon'] == str(recon)][str(i)].quantile(q=0.25))
            # combine statistics from each Recon method into dataframe
            df_stats = pd.DataFrame({'Recon': recon_list, 'Mean': df[df['Recon'] == str(recon)][str(i)].mean(), 'Stdev': df[df['Recon'] == str(recon)][str(i)].std(
            ), 'Median': df[df['Recon'] == str(recon)][str(i)].median(), 'IQR': df[df['Recon'] == str(recon)][str(i)].quantile(q=0.75)-df[df['Recon'] == str(recon)][str(i)].quantile(q=0.25)})
            # combine df_stats from each Recon method into dataframe
            df_stats_all = pd.concat([df_stats, df_stats_all], axis=0)
        print(df_stats_all)
        # save dataframe to csv
        df_stats_all.to_csv(main_path+'/'+file.stem +
                            '_walk_length_'+str(i)+'_stats.csv', index=False)
        # calculate p-value for difference in score between Recon methods in recon_list, with comparisons between each pair of Recon methods, save all statistics and p-values to csv
        df_ttest_results = pd.DataFrame()
        df_ttest_rel_results = pd.DataFrame()
        df_z_ttest_results = pd.DataFrame()
        df_z_ttest_rel_results = pd.DataFrame()
        for recon in recon_list:
            for recon2 in recon_list:
                if recon != recon2:
                    print(recon)
                    print(recon2)
                    # calculate t-test on Pearson scores, get t-statistic and p-value
                    ttest_pearson_score = stats.ttest_ind(df[df['Recon'] == str(
                        recon)][str(i)], df[df['Recon'] == str(recon2)][str(i)], alternative='greater')
                    ttest_rel_pearson_score = ttest_rel(df[df['Recon'] == str(
                        recon)][str(i)], df[df['Recon'] == str(recon2)][str(i)], alternative='greater')
                    ttest_z_score = stats.ttest_ind(df_z[df_z['Recon'] == str(
                        recon)][str(i)], df_z[df_z['Recon'] == str(recon2)][str(i)], alternative='greater')
                    ttest_rel_z_score = ttest_rel(df_z[df_z['Recon'] == str(
                        recon)][str(i)], df_z[df_z['Recon'] == str(recon2)][str(i)], alternative='greater')
                    # combine t-test results into dataframe
                    df_ttest_results = pd.concat([df_ttest_results, pd.DataFrame({'Recon 1': recon, 'Recon 2': recon2, 'T-statistic Pearson Score t-test': ttest_pearson_score[0],
                                                 'p-value Pearson Score t-test':ttest_pearson_score[1]}, index=[0])], axis=0)  # ,'T-statistic z-score':ttest_z_score[0],'p-value z-score':ttest_z_score[1]
                    df_ttest_rel_results = pd.concat([df_ttest_rel_results, pd.DataFrame({'Recon 1': recon, 'Recon 2': recon2, 'T-statistic Pearson Score t-test': ttest_rel_pearson_score[0],
                                                    'p-value Pearson Score t-test':ttest_rel_pearson_score[1]}, index=[0])], axis=0)  # ,'T-statistic z-score':ttest_z_score[0],'p-value z-score':ttest_z_score[1]
                    df_z_ttest_rel_results = pd.concat([df_z_ttest_rel_results, pd.DataFrame({'Recon 1': recon, 'Recon 2': recon2, 'T-statistic z-score t-test': ttest_rel_z_score[0],
                                                    'p-value z-score t-test':ttest_rel_z_score[1]}, index=[0])], axis=0)  # ,'T-statistic z-score':ttest_z_score[0],'p-value z-score':ttest_z_score[1]
                    df_z_ttest_results = pd.concat([df_z_ttest_results, pd.DataFrame({'Recon 1': recon, 'Recon 2': recon2, 'T-statistic z-score t-test': ttest_z_score[0],
                                                    'p-value z-score t-test':ttest_z_score[1]}, index=[0])], axis=0)  # ,'T-statistic z-score':ttest_z_score[0],'p-value z-score':ttest_z_score[1]
        # print('T-test ind results')
        # print(df_ttest_results)
        # print('T-test rel results')
        # print(df_ttest_rel_results)
        # print('T-test ind z-score results')
        # print(df_z_ttest_results)
        # print('T-test rel z-score results')
        # print(df_z_ttest_rel_results)

        # save dataframe to csv
        df_ttest_results.to_csv(
            main_path+'/'+file.stem+'_walk_length_'+str(i)+'_ttest_results.csv', index=False)
        df_ttest_rel_results.to_csv(
            main_path+'/'+file.stem+'_walk_length_'+str(i)+'_ttest_rel_results.csv', index=False)
        df_z_ttest_results.to_csv(
            main_path+'/'+file.stem+'_walk_length_'+str(i)+'_z_score_ttest_results.csv', index=False)
        df_z_ttest_rel_results.to_csv(
            main_path+'/'+file.stem+'_walk_length_'+str(i)+'_z_score_ttest_rel_results.csv', index=False)

file_list = Path(main_path).glob('*batch??.csv')
for file in file_list:
    print(file)
    df = pd.read_csv(file)
    print(file.stem)
    # melt data for boxplot
    dd=pd.melt(df,id_vars=['Recon'],value_vars=['1','2','3','4','5'],var_name='Walk Length')
    # seaborn boxplot with hue based on recon method
    sns.boxplot(x='Walk Length',y='value',data=dd,hue='Recon')
    # plt.show()
    plt.ylabel('Pearson Score')
    # save figure
    plt.savefig(main_path+'/'+file.stem+'_box_plot.png')
    plt.close()