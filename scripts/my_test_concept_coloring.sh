#!/bin/sh

python ../train_places.py ConceptWhitening/data/data_256/ \
    --seed 1232 \
    --ngpu 1 \
    --workers 60 \
    --arch resnet_cw \
    --depth 18 \
    --epochs 200 \
    --batch-size 1 \
    --lr 0.1 \
    --whitened_layers 7 \
    --concepts airplane,bicycle,car,train,dining_table,microwave,oven,cat,dog \
    --prefix RESNET18_PLACES365_TRANSFER_CC \
    --evaluate /
