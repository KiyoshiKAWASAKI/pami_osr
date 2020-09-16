#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

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


debugging_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/valid.json"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize])

train_dataset = msd_net_dataset(json_path=debugging_json_path,
                                transform=train_transform)
train_set_index = torch.randperm(len(train_dataset))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=2,
                                           shuffle=False,
                                           sampler=torch.utils.data.RandomSampler(train_dataset),
                                           collate_fn=customized_dataloader.collate)

for i, batch in enumerate(train_loader):
    print("@"*30)
    print(i)
    # print(batch)

















