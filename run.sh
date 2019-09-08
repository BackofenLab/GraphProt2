#!/bin/bash

# input arguments
mode=2 # 0: cross validation, 1: holdout, 2: generic
dataset_name=GENERIC
patience=3
batch_size=50
epochs=10
fc_hidden_dim=128
kfsplits=3
list_lr='0.0001' # '0.001 0.0005 0.0001'
list_node_hidden_dim='128' # '64 96 128'
list_weight_decay='0.0001 0.00001'


if [ ${mode} == 0 ]; then
    python main_cv.py \
        --dataset $dataset_name \
        --patience $patience \
        --batch_size $batch_size \
        --epochs $epochs \
        --fc_hidden_dim $fc_hidden_dim \
        --kfsplits $kfsplits \
        --list_lr $list_lr \
        --list_node_hidden_dim $list_node_hidden_dim \
        --list_weight_decay $list_weight_decay \
        --use_node_attr

elif [ ${mode} == 1 ]; then
    python main_holdout.py \
        --dataset $dataset_name \
        --patience $patience \
        --batch_size $batch_size \
        --epochs $epochs \
        --fc_hidden_dim $fc_hidden_dim \
        --list_lr $list_lr \
        --list_node_hidden_dim $list_node_hidden_dim \
        --list_weight_decay $list_weight_decay \
        --use_node_attr
else
    python main_generic.py \
      --dataset $dataset_name \
      --patience $patience \
      --batch_size $batch_size \
      --epochs $epochs \
      --fc_hidden_dim $fc_hidden_dim \
      --list_lr $list_lr \
      --list_node_hidden_dim $list_node_hidden_dim \
      --list_weight_decay $list_weight_decay \
      --use_node_attr
fi