#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -e errors/
#$ -N ce_seed_0

# Required modules
module load conda
conda init bash
source activate new_msd_net

python train_known_only.py --arch msdnet \
                           --batch-size 16 \
                           --nb_training_classes 294 \
                           --optimizer sgd \
                           --learning-rate 0.1 \
                           --epochs 200 \
                           --test_with_novel True \
                           --print-freq 1