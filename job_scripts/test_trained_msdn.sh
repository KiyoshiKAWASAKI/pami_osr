#!/bin/bash
# Required modules
module load conda
conda init bash
source activate MSDNet

python -W ignore main.py \
                --data-root /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1/known_classes/images \
                --test_folder_name val \
                --evaluate-from /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/save_models/checkpoint_099.pth.tar \
                --save /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/test_results_0818 \
                --save_probs_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/test_results_0818/probs.npy \
                --save_targets_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/test_results_0818/targets.npy \
                --save_original_label_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/test_results_0818/original_labels.npy \
                --save_rt_path /afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/msd_369_novel_44_weighted_loss_simple_setup2/test_results_0818/rts.npy \
                --epochs 1 \
                --batch-size 1 \
                --thresh_top_1 0.90 \
                --thresh_top_3 0.05 \
                --thresh_top_5 0.03 \
                --evalmode anytime \
                --nb_training_classes 369 \
                --test_with_novel True \
                --save_probs True \
                --gpu 2 \
                --arch msdnet \
                --use-valid