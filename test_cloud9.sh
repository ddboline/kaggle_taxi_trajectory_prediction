#!/bin/bash

# for F in train.csv.gz test.csv.gz metaData_taxistandsID_name_GPSlocation.csv.gz;
# do
#     scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_taxi_trajectory_prediction/$F .
# done

for F in train_idx.csv.gz train_nib.csv.gz test_idx.csv.gz test_nib.csv.gz test_trj.csv.gz sampleSubmission.csv.gz;
do
    scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_taxi_trajectory_prediction/$F .
done
mkdir -p train
scp ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_taxi_trajectory_prediction/train/train_trj_*.csv.gz train/

# time ./feature_extraction.py
# ./load_data.py $1
# ./my_model.py

# D=`date +%Y%m%d%H%M%S`
# tar zcvf output_${D}.tar.gz model.pkl.gz output.out output.err
# scp model_*.pkl.gz ddboline@ddbolineathome.mooo.com:/home/ddboline/setup_files/build/kaggle_taxi_trajectory_prediction/
ssh ddboline@ddbolineathome.mooo.com "~/bin/send_to_gtalk done_kaggle_taxi_trajectory_prediction"
echo "JOB DONE taxi_trajectory_prediction"
