import os
import time
import torch
from torchvision import datasets, transforms
import customized_dataloader
from customized_dataloader import msd_net_with_grouped_rts
from customized_dataloader import msd_net_dataset
import warnings
warnings.filterwarnings("ignore")

##########################################################################
# Json file paths
##########################################################################
train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                             "npy_json_files/rt_group_json/valid_known_unknown.json"

old_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                "npy_json_files/valid_known_unknown.json"


##########################################################################
# Data transforms
##########################################################################
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      normalize])

# Old data loader
tkk_dataset_old = msd_net_dataset(json_path=old_json_path,
                                 transform=train_transform)
tkk_index = torch.randperm(len(tkk_dataset_old))
tkk_loader_old = torch.utils.data.DataLoader(tkk_dataset_old,
                                           batch_size=16,
                                           shuffle=False,
                                           sampler=torch.utils.data.RandomSampler(tkk_index),
                                           drop_last=True,
                                           collate_fn=customized_dataloader.collate)

# Training loaders
train_known_known_dataset = msd_net_with_grouped_rts(json_path=train_known_known_path,
                                                    transform=train_transform)
train_known_known_index = torch.randperm(len(train_known_known_dataset))

train_known_known_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                                       batch_size=1,
                                                       shuffle=False,
                                                       collate_fn=customized_dataloader.collate_new,
                                                       sampler=torch.utils.data.RandomSampler(train_known_known_index),
                                                       drop_last=True)

# Test and compare the 2 data loaders
# batch = next(iter(tkk_loader_old))
new_batch = next(iter(train_known_known_loader))
