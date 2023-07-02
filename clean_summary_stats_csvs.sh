#!/bin/bash
# ./clean_summary_stats_csvs.sh /path/to/prediction_folder
predict_path=$1
for i in `seq 0 49`; do ii="${predict_path}/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done
for i in `seq 0 49`; do ii="${predict_path}/dataset/stats/mean_path_length_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done
for i in `seq 0 49`; do ii="${predict_path}/dataset/stats/volume_weighted_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done


