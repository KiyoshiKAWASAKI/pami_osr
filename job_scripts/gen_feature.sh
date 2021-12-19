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

python generate_features.py --arch msdnet \
                            --batch-size 1 \
                            --generate_feature True \
                            --nb_training_classes 296 \
                            --test_with_novel True \
                            --print-freq 1