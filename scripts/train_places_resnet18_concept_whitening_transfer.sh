#!/bin/bash
python3 ../train_imagenet.py --ngpu 1 --workers 4 --arch resnet_transfer --depth 18 --epochs 200 --batch-size 64 --lr 0.05 --whitened_layers 5 --concepts airplane,bed,person --prefix RESNET18_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
