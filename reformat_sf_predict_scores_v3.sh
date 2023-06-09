#!/bin/bash
#
# reformat_sf_predict_scores_v3.sh
#
# from parent directory containing dataset folder!
#
# ./reformat_sf_predict_scores_v3.sh

scoreFiles=./dataset/scores*csv

for ii in $scoreFiles;
do
	if [[ "$ii" == *"clean"* ]]; then
  		echo "It's there!"
	else
		NUM=`echo "${ii::-4}" | rev | cut -d_ -f1 | rev | cut -dn -f2`
		if [ ${#NUM} == 1 ];
		then
    			SETNUM="00${NUM}"
		elif [ ${#NUM} == 2 ];
		then
			SETNUM="0${NUM}"
		elif [ ${#NUM} == 3 ];
		then
			SETNUM="${NUM}"
		fi
		PREFIX=`echo ${ii} | rev | cut -d_ -f1 --complement | rev`
		NEW="${PREFIX}_${SETNUM}_clean.csv"
		echo "Participant_Number,Pearson_Walk_1,Pearson_Walk_2,Pearson_Walk_3,Pearson_Walk_4,Pearson_Walk_5,Pearson_Walk_6,Pearson_Walk_7,Pearson_Walk_8,Pearson_Walk_9,Pearson_Walk_10" > ${NEW}_tmp
		tail -n 21 ${ii} >> ${NEW}_tmp
                echo "Batch_Number" > batches.csv
                for ii in {1..21}; do echo ${SETNUM}; done >> batches.csv
                paste -d, ${NEW}_tmp batches.csv > ${NEW}
                rm ${NEW}_tmp
	fi
done

