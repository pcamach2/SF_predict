for i in `seq 0 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done


for i in `seq 0 1`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii}; done

for i in `seq 2 46`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii}; done
for i in `seq 48 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii}; done

/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/count_all_percent_batch${i}_walk_length_4_stats.csv




for i in `seq 0 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/volume_weighted_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 5 ${ii} | tail -n 1 >> tmp; mv tmp ${ii}_clean.csv; done

for i in `seq 0 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/mean_path_length_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 5 ${ii} | tail -n 1 >> tmp; mv tmp ${ii}_clean.csv; done


for i in `seq 0 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/mean_path_length_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done
for i in `seq 0 49`; do ii="/home/paul/thesis/dev/SAY_sf_prediction_v4/dataset/stats/volume_weighted_all_percent_batch${i}_walk_length_4_stats.csv"; echo ${ii}; head -n 1 ${ii} > tmp; tail -n 3 ${ii} | head -n 1 >> tmp; head -n 6 ${ii} | tail -n 1 >> tmp; head -n 4 ${ii} | tail -n 1 >> tmp; mv tmp ${ii::-4}_clean.csv; done


