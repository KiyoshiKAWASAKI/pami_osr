#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -e errors/
#$ -N feat_14

# Required modules
module load conda
conda init bash
source activate new_msd_net

python obtain_init_threshold.py --arch msdnet \
                             --batch-size 16 \
                             --generate_feature True \
                             --nb_training_classes 294 \
                             --test_with_novel True \
                             --print-freq 1