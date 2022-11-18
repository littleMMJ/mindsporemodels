#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train.sh DATA_PATH TRAIN_PATH DEVICE_ID"
echo "For example: bash run_standalone_train.sh ./BraST17/HGG ./src/train.txt 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
if [ $# != 3 ]
then
  echo "Usage: bash run_standalone_train.sh DATA_PATH TRAIN_PATH DEVICE_ID"
exit 1
fi

DATA_PATH=$1
TRAIN_PATH=$2
DEVICE_ID=$3

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..
env > env.log
echo "Train begin"
python train.py \
--data_path "$DATA_PATH" \
--train_path "$TRAIN_PATH" \
--correction True \
--device_id "$DEVICE_ID" \
> train.log 2>&1 &
echo "Training. Check it at train.log"
