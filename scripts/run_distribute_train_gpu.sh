#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train_gpu.sh"
echo "For example: bash sh run_distribute_train_gpu.sh MINDRECORD_PATH CONFIG_PATH"
echo "=============================================================================================================="
set -e
if [ $# != 2 ]
then
  echo "Usage: bash run_distribute_train_gpu.sh MINDRECORD_PATH CONFIG_PATH"
exit 1
fi

MINDRECORD_PATH=$1
CONFIG_PATH=$2


export MINDRECORD_PATH=${MINDRECORD_PATH}
export CONFIG_PATH=${CONFIG_PATH}


echo "distribute train begin."
cd ../
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 mpirun --allow-run-as-root -n 8 \
python train.py > train_distributed.log 2>&1 \
--mindrecord_path=$MINDRECORD_PATH \
--config_path=$CONFIG_PATH \
--run_distribute=True &
echo "Training background..."
