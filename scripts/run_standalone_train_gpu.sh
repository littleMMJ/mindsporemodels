#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train_gpu.sh"
echo "For example: bash run_standalone_train_gpu.sh MINDRECORD_PATH CONFIG_PATH DEVICE_ID"
echo "=============================================================================================================="
set -e
if [ $# != 3 ]
then
  echo "Usage: bash run_standalone_train_gpu.sh MINDRECORD_PATH CONFIG_PATH DEVICE_ID"
exit 1
fi

MINDRECORD_PATH=$1
CONFIG_PATH=$2
DEVICE_ID=$3


export MINDRECORD_PATH=${MINDRECORD_PATH}
export CONFIG_PATH=${CONFIG_PATH}
export DEVICE_ID=${DEVICE_ID}

echo "Standalone train begin."
cd ../
nohup python train.py > train_standalone.log 2>&1 \
--mindrecord_path=$MINDRECORD_PATH \
--config_path=$CONFIG_PATH \
--device_id=$DEVICE_ID \
--run_distribute=False &

echo "Training background..."
