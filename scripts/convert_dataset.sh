#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash convert_dataset.sh"
echo "For example: sh convert_dataset.sh DATA_PATH TRAIN_PATH MINDRECORD_PATH SAMPLE_NUM"
echo "=============================================================================================================="
set -e
if [ $# != 4 ]
then
  echo "Usage: bash convert_dataset.sh DATA_PATH TRAIN_PATH MINDRECORD_PATH SAMPLE_NUM"
exit 1
fi

DATA_PATH=$1
TRAIN_PATH=$2
MINDRECORD_PATH=$3
SAMPLE_NUM=$4

export DATA_PATH=${DATA_PATH}
export TRAIN_PATH=${TRAIN_PATH}
export MINDRECORD_PATH=${MINDRECORD_PATH}
export SAMPLE_NUM=${SAMPLE_NUM} # recommend 400

if [ ! -d "$DATA_PATH" ]; then
        echo "dataset does not exit"
        exit
fi

if [ ! -d "$MINDRECORD_PATH" ]; then
        mkdir $MINDRECORD_PATH
        echo "mindrecord path create success"
fi

echo "mindrecord_generator begin."
cd ../src/
nohup python mindrecord_generator.py > dataset.log 2>&1 \
--data-path=$DATA_PATH \
--train-path=$TRAIN_PATH \
--mindrecord-path=$MINDRECORD_PATH \
--sample-num=$SAMPLE_NUM &
echo "mindrecord_generator background..."
