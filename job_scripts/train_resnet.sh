#!/bin/bash

#$ -M jhuang24@nd.edu
#$ -m abe
#$ -q gpu -l gpu=1
#$ -l h=!qa-rtx6k-044
#$ -l h=!qa-a10-001
#$ -l h=!qa-a10-002
#$ -e errors/
#$ -N resnet_psyphy_seed_0

# Required modules
module load conda
conda init bash
source activate new_msd_net

python pipeline_openset_resnet.py