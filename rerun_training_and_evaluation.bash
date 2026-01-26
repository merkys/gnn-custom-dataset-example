#!/bin/bash

cd "$(dirname $0)"

echo
echo "Clearing previously generated files ..."

rm -rf ./training_data ./validation_data ./testing_data ./outputs/molecules-bond-sum ./__pycache__

echo
echo "Running training using 'training_data' ..."

python run_training.py

echo
echo "Running evaluation using 'training_data' ..."

find "./outputs/molecules-bond-sum/" -type f | sort -V | xargs python ./run_evaluation.py training_data

echo
echo "Running evaluation using 'validation_data' ..."

find "./outputs/molecules-bond-sum/" -type f | sort -V | xargs python ./run_evaluation.py validation_data

echo
echo "Running evaluation using 'testing_data' ..."

find "./outputs/molecules-bond-sum/" -type f | sort -V | xargs python ./run_evaluation.py testing_data

