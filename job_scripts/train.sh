#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N msd_pp_add_s_3.0

# Required modules
module load conda
conda init bash
source activate new_msd_net

CUDA_VISIBLE_DEVICES=3 python demo.py --arch msdnet \
                                       --batch-size 16 \
                                       --nb_training_classes 296 \
                                       --optimizer sgd \
                                       --learning-rate 0.1 \
                                       --gpu 1 \
                                       --epochs 200 \
                                       --print-freq 1