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

from customized_dataloader import msd_net_dataset
import customized_dataloader


debugging_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_unknown.json"


# TODO: debugging for getting dataset and
train_dataset = msd_net_dataset(json_path=debugging_json_path)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=1,
                                  collate_fn=customized_dataloader.collate)

for i, batch in enumerate(train_loader):
    if i <= 10:
        print(i)
        print(batch)

    else:
        break

















