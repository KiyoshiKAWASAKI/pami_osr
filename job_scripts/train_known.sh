#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -l h=!qa-xp-012
#$ -e errors/
#$ -N known_only_pp_1.0_seed_4

# Required modules
module load conda
conda init bash
source activate new_msd_net

python known_pipeline.py --arch msdnet \
                         --batch-size 16 \
                         --nb_training_classes 293 \
                         --optimizer sgd \
                         --learning-rate 0.1 \
                         --epochs 200 \
                         --test_with_novel True \
                         --print-freq 1