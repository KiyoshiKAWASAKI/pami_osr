#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -N debug_set_5_weights_200_epoch

# Required modules
module load conda
conda init bash
source activate new_msd_net

# Application to execute
python -W ignore main.py --train_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                           --train_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                           --valid_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                           --valid_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                           --test_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json \
                           --test_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                           --test_unknown_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json \
                           --train_early_exit True \
                           --arch msdnet \
                           --batch-size 32 \
                           --nb_training_classes 336 \
                           --switch_batch True \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1101/debug_set_5_weights_200_epoch \
                           --optimizer sgd \
                           --learning-rate 0.1 \
                           --epochs 200 \
                           --print-freq 1