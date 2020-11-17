#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N dense_net_old_loader

# Required modules
module load conda
conda init bash
source activate new_msd_net

# Application to execute
python -W ignore dense_net_old_loader.py --data-root /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_baseline_0722/small_data_for_debug \
                                       --train_folder_name train \
                                       --test_folder_name val \
                                       --batch-size 16 \
                                       --nb_training_classes 335 \
                                       --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/dense_net/debug_1104_all/use_old_loader \
                                       --use-valid \
                                       --arch msdnet \
                                       --nBlocks 5 \
                                       --optimizer sgd \
                                       --learning-rate 0.1 \
                                       --gpu 1 \
                                       --epochs 50 \
                                       --print-freq 1