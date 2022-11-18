#!/bin/bash
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_distribute_train.sh DATA_PATH TRAIN_PATH RANK_TABLE_FILE"
echo "For example: bash run_distribute_train.sh ./BraST17/HGG ./src/train.txt ./rank_table_8pcs.json"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
DATA_PATH=$1
export DATA_PATH=${DATA_PATH}
TRAIN_PATH=$2
export TRAIN_PATH=${TRAIN_PATH}
RANK_TABLE_FILE=$3
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export HCCL_CONNECT_TIMEOUT=6000

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
    cd ./device$i
    mkdir src
    cd ../
    cp ../train.py ./device$i
    cp ../*.yaml ./device$i
    cp ../src/*.py ./device$i/src
    cp ../src/*.txt ./device$i/src
    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > env$i.log
    python train.py --data_path "$DATA_PATH" --train_path "$TRAIN_PATH" --correction True --run_distribute True > train$i.log 2>&1 &
    echo "$i finish"
    cd ../
done
rm -rf device0
mkdir device0
cd ./device0
mkdir src
cd ../
cp ../train.py ./device0
cp ../*.yaml ./device0
cp ../src/*.py ./device0/src
cp ../src/*.txt ./device0/src
cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > env0.log
python train.py --data_path "$DATA_PATH" --train_path "$TRAIN_PATH" --correction True --run_distribute True > train0.log 2>&1 &
if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
cd ../