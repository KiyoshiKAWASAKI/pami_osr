#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N test_train_valid_pp_add

# Required modules
module load conda
conda init bash
source activate new_msd_net

CUDA_VISIBLE_DEVICES=0g python demo.py --arch msdnet \
                                       --batch-size 1 \
                                       --nb_training_classes 296 \
                                       --optimizer sgd \
                                       --learning-rate 0.1 \
                                       --gpu 1 \
                                       --epochs 100 \
                                       --test_with_novel True \
                                       --print-freq 1