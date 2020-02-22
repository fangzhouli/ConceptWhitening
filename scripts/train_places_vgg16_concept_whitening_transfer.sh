#!/bin/bash
python3 ../train_imagenet.py --ngpu 1 --workers 4 --arch vgg16_bn_transfer --depth 16 --epochs 200 --batch-size 64 --lr 0.1 --whitened_layers 13 --concepts airplane,bed,person --prefix VGG16_PLACES365_CPT_WHITEN_TRANSFER /usr/xtmp/zhichen/data_256/
