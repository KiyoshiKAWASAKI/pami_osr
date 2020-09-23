#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import warnings
warnings.filterwarnings("ignore")

# import os
# import sys
# import math
# import time
# import shutil
# import numpy as np
# import logging
# import csv
#
# from args import arg_parser
# from adaptive_inference import dynamic_evaluate
# import models
# from op_counter import measure_model
# from itertools import islice

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from timeit import default_timer as timer
import datetime
import torchvision.transforms as transforms

from customized_dataloader import msd_net_dataset
import customized_dataloader

train_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train.json"
valid_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid.json"



def test_data_loader(json_path,
                     check_data=False):
    """
    1. Use this function to test whether the data loader is working
    2. Go thru all the data to check whether the shapes are correct
    :param json_path:
    :return:
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        normalize])

    dataset = msd_net_dataset(json_path=json_path,
                              transform=transform)
    set_index = torch.randperm(len(dataset))

    if check_data:
        print("Checking data...")
        data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=32,
                                                   shuffle=False,
                                                   collate_fn=customized_dataloader.collate)

    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=32,
                                                   shuffle=False,
                                                   sampler=torch.utils.data.RandomSampler(set_index),
                                                   collate_fn=customized_dataloader.collate)


    for i, batch in enumerate(data_loader):
        print(i)



if __name__ == '__main__':
    test_data_loader(json_path=train_json_path,
                     check_data=True)
    # test_data_loader(json_path=valid_json_path,
    #                  check_data=True)


















