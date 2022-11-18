#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh DATA_PATH TEST_PATH CKPT_PATH DEVICE_ID"
echo "For example: bash run_eval.sh ./BraST17/HGG ./src/test.txt ./*.ckpt 0"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ $# != 4 ]
then
  echo "Usage: bash run_eval.sh DATA_PATH TEST_PATH CKPT_PATH DEVICE_ID"
exit 1
fi

DATA_PATH=$1
TEST_PATH=$2
CKPT_PATH=$3
DEVICE_ID=$4

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"
export DEVICE_ID=$DEVICE_ID
export RANK_ID=$DEVICE_ID
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd ..
env > env.log
echo "Eval begin"
python eval.py \
--data_path "$DATA_PATH" \
--test_path "$TEST_PATH" \
--ckpt_path "$CKPT_PATH" \
--correction True \
--device_id "$DEVICE_ID" \
> eval.log 2>&1 &
echo "Evaling. Check it at eval.log"