#!/bin/sh

#SBATCH --job-name=cc-test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/share/taglab/Fang/concept-coloring-preliminary/logs/%j.out
#SBATCH --error=/share/taglab/Fang/concept-coloring-preliminary/logs/%j.err
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=cuda:6.1
#SBATCH --mem=300G
#SBATCH --time=3:00:00

## Timestmaps: 20210903_073210
##             20210904_115816
##             20210904_205429
##             20210905_131108

python ../train_places.py /share/taglab/Fang/cw/data_256/ \
    --seed 1232 \
    --ngpu 1 \
    --workers 32 \
    --arch resnet_cw \
    --depth 18 \
    --epochs 200 \
    --batch-size 1 \
    --lr 0.1 \
    --whitened_layers 7 \
    --concepts airplane,bicycle,car,train,dining_table,microwave,oven,cat,dog \
    --checkpoint-timestamp 20210903_073210 \
    --prefix RESNET18_PLACES365_TRANSFER_CC \
    --evaluate
