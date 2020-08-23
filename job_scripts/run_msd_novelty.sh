#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu@@cvrl-titanxp -l gpu=1
#$ -N known_369_unknown_44_simple_weight_0813

# Required modules
module load conda
conda init bash
source activate MSDNet

# Application to execute
python -W ignore main.py --data-root /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_baseline_0722/known_classes/images/ \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2 \
                           --log_file_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss/0813_known_369_unknown_44_simple_weight.log \
                           --train_folder_name train \
                           --test_folder_name test_known_369 \
                           --train_early_exit True \
                           --arch msdnet \
                           --batch-size 128 \
                           --nb_training_classes 369 \
                           --epochs 100 \
                           --use-valid \
                           --gpu 3

