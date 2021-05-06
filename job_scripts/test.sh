#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu@@cvrl-titanxp -l gpu=1
#$ -N run_train_valid_1

# Required modules
module load conda
conda init bash
source activate new_msd_net

python demo.py --arch msdnet \
               --batch-size 1 \
               --nb_training_classes 296 \
               --optimizer sgd \
               --learning-rate 0.1 \
               --epochs 100 \
               --test_with_novel True \
               --print-freq 1