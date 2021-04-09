#!/bin/bash

dataset=$1
option=$2  # savef / loadf
declare -a devices=($3 $4 $5 $6 $7)

mkdir -p logs ckpts

if [ $dataset == "PROTEINS" ]
then
    args="--epochs 20 --lr 0.0005 --hdim 256 --pooling_ratio 0.54 --lamb 1.26 --weight_decay 0.0001"
elif [ $dataset == "NCI1" ]
then
    args="--epochs 50 --lr 0.001 --hdim 512 --pooling_ratio 0.62 --lamb 0.82 --weight_decay 0.001"
elif [ $dataset == "MUTAG" ]
then
    args="--epochs 100 --lr 0.0005 --hdim 512 --pooling_ratio 0.24 --lamb 1.4 --weight_decay 0.0001"
elif [ $dataset == "Mutagenicity" ]
then
    args="--epochs 100 --lr 0.001 --hdim 512 --pooling_ratio 0.535 --lamb 0.713 --weight_decay 0.00001"
else
    echo Unsupported dataset
    exit 0
fi

cnt=1
for device in "${devices[@]}"
do
    python -u run.py --dataset $dataset --device cuda:$device --$option ckpts/${dataset}_${cnt}.pt $args | tee logs/${option}_${dataset}_${cnt}.log &
    ((cnt++))
done
