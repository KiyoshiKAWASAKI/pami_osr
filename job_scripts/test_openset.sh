#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -N test_00

# Required modules
module load conda
conda init bash
source activate new_msd_net

python pipeline.py --arch msdnet \
               --batch-size 1 \
               --nb_training_classes 293 \
               --optimizer sgd \
               --learning-rate 0.1 \
               --epochs 100 \
               --test_with_novel True \
               --print-freq 1