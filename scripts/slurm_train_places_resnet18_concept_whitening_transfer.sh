#!/bin/sh

#SBATCH --job-name=cc
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fzli@ucdavis.edu
#SBATCH --output=/share/taglab/Fang/concept-coloring-preliminary/logs/%j.out
#SBATCH --error=/share/taglab/Fang/concept-coloring-preliminary/logs/%j.err
#SBATCH --partition=production
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=cuda:6.0
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00

python ../train_places.py --ngpu 1 --workers 32 --arch resnet_cw --depth 18 --epochs 200 --batch-size 64 --lr 0.05 --whitened_layers 7 --concepts airplane,bicycle,car,train,dining_table,microwave,oven,cat,dog --prefix RESNET18_PLACES365_TRANSFER_CC /share/taglab/Fang/cw/data_256/
