#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_gpu.sh "
echo "For example: bash run_eval_gpu.sh DATA_PATH TEST_PATH CKPT_PATH CONFIG_PATH DEVICE_ID"
echo "=============================================================================================================="

set -e
if [ $# != 5 ]
then
  echo "Usage: bash run_eval_gpu.sh DATA_PATH TEST_PATH CKPT_PATH CONFIG_PATH DEVICE_ID"
exit 1
fi

DATA_PATH=$1
TEST_PATH=$2
CKPT_PATH=$3
CONFIG_PATH=$4
DEVICE_ID=$5


export DATA_PATH=${DATA_PATH}
export TEST_PATH=${TEST_PATH}
export CKPT_PATH=${CKPT_PATH}
export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_ID=${DEVICE_ID}

if [ ! -d "$DATA_PATH" ]; then
        echo "dataset does not exit"
        exit
fi

echo "eval begin."
cd ../
nohup python eval.py > eval.log 2>&1 \
--data_path=$DATA_PATH \
--test_path=$TEST_PATH \
--ckpt_path=$CKPT_PATH \
--config_path=$CONFIG_PATH \
--device_id=$DEVICE_ID &
echo "eval background..."
