#!/bin/bash

# Required modules
module load conda
conda init bash
source activate new_msd_net

# Application to execute
python -W ignore main.py --train_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known.json \
                           --train_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown.json \
                           --valid_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known.json \
                           --valid_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown.json \
                           --test_known_known_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known.json \
                           --test_known_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown.json \
                           --test_unknown_unknown_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown.json \
                           --train_early_exit True \
                           --arch msdnet \
                           --batch-size 32 \
                           --use-valid \
                           --nb_training_classes 336 \
                           --gpu 1 \
                           --use_5_weights False \
                           --use_pp_loss False \
                           --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_debug_set_all/ce_sgd_lr_0.1 \
                           --tf_board_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_debug_set_all/ce_sgd_lr_0.1/vis \
                           --optimizer sgd \
                           --log_file_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/3_part_pp_loss/1004_debug_set_all/ce_sgd_lr_0.1/record.log \
                           --learning-rate 0.1 \
                           --epochs 20 \
                           --print-freq 1

