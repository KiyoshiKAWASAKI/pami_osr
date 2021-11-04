# Generate features for MSD-Net and ResNet


import os
import time
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
# from models import resnet
import numpy as np
from timeit import default_timer as timer
from utils import customized_dataloader
from utils.customized_dataloader import msd_net_dataset, msd_net_with_grouped_rts
import sys
import warnings
import torchvision

warnings.filterwarnings("ignore")
import random
from args import arg_parser
import torch.nn as nn
import models
from datetime import datetime
import math

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

date = datetime.today().strftime('%Y-%m-%d')
model_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/"

###################################################################
# Change these for each model
###################################################################
# model_name = "msd_net"
model_name = "resnet_50"
model_used = "model_epoch_34.dat"
model_dir = "cvpr_resnet/2021-10-24/resnet_50_seed_0/"
test_model_path = model_path_base + model_dir + model_used

debug = False
epoch_index = None

####################################################################
# Normally, there is no need to change these #
####################################################################
nb_itr = 30
nb_clfs = 5
batch_size = 1
img_size = 224
nBlocks = 5
nb_classes = 296
nb_training_classes = 296


#########################################################################################
# Define paths for data source #
#########################################################################################
train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                         "dataset_v1_3_partition/npy_json_files/train_known_known.json"
train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                           "dataset_v1_3_partition/npy_json_files/train_known_unknown.json"

test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                        "dataset_v1_3_partition/npy_json_files/test_known_known_without_rt.json"
test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                          "dataset_v1_3_partition/npy_json_files/test_known_unknown.json"
test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                            "dataset_v1_3_partition/npy_json_files/test_unknown_unknown.json"


##########################################################################
# Define all the functions #
##########################################################################
def gen_feature(loader,
                model,
                use_msd_net,
                data_category,
                save_dir):
    """
    For the labels:
        We set the labels for known_unknowns as 0 to match with EVM.
        Labels for known_knowns start from 1.

    :param test_loader:
    :param model:
    :param test_unknown:
    :param use_msd_net:
    :return:
    """
    # Set the model to evaluation mode
    model.cuda()
    model.eval()

    full_original_label_list = []
    full_feature_list = []


    for i in range(len(loader)):
        batch = next(iter(loader))

        input = batch["imgs"]

        if (data_category == "train_known_unknown") or (data_category == "test_known_unknown"):
            target = 0
        else:
            target = batch["labels"]

        input = input.cuda()
        target = target.cuda(async=True)

        # Save original labels to the list
        original_label_list = np.array(target.cpu().tolist())
        for label in original_label_list:
            full_original_label_list.append(label)

        input_var = torch.autograd.Variable(input)

        # Get the features from model
        if model_name == "msd_net":
            print("Generating feature for MSD-Net.")
            output, feature, end_time = model(input_var)

            feature = feature[0][0].cpu().detach().numpy()


        elif model_name == "resnet_50":
            # model.replace_logits
            # TODO: nb classes in incorrect here, it should be 296 not 1000
            output, feature = model(input_var)

            feature = feature.cpu().detach().numpy()

        # TODO: Add other resnet arch
        else:
            pass

        feature = np.reshape(feature, (1, feature.shape[1] * feature.shape[2] * feature.shape[3]))

        if i == 0:
            full_feature_list = feature
        else:
            full_feature_list = np.concatenate((full_feature_list, feature), axis=0)

        print(full_feature_list.shape)
        print(len(full_original_label_list))

    # Save all results to npy
    full_label_list = np.array(full_original_label_list)

    save_label_dir = save_dir + data_category + "_labels.npy"
    save_feature_dir = save_dir + data_category + "_features.npy"

    np.save(save_label_dir, full_label_list)
    np.save(save_feature_dir, full_feature_list)

    print("NPY files saved!")




def demo(depth=100,
         growth_rate=12,
         efficient=True):

    global args

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    #######################################################################
    # Create dataset and data loader
    #######################################################################
    train_known_known_dataset = msd_net_dataset(json_path=train_known_known_path,
                                                transform=train_transform)
    train_known_unknown_dataset = msd_net_dataset(json_path=train_known_unknown_path,
                                                  transform=train_transform)

    train_known_known_index = torch.randperm(len(train_known_known_dataset))
    train_known_unknown_index = torch.randperm(len(train_known_unknown_dataset))

    train_known_known_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           collate_fn=customized_dataloader.collate,
                                                           sampler=torch.utils.data.RandomSampler(
                                                               train_known_known_index))
    train_known_unknown_loader = torch.utils.data.DataLoader(train_known_unknown_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             drop_last=True,
                                                             collate_fn=customized_dataloader.collate,
                                                             sampler=torch.utils.data.RandomSampler(
                                                                 train_known_unknown_index))

    # Test loaders
    test_known_known_dataset = msd_net_dataset(json_path=test_known_known_path,
                                               transform=test_transform)
    test_known_known_index = torch.randperm(len(test_known_known_dataset))

    test_known_unknown_dataset = msd_net_dataset(json_path=test_known_unknown_path,
                                                 transform=test_transform)
    test_known_unknown_index = torch.randperm(len(test_known_unknown_dataset))

    test_unknown_unknown_dataset = msd_net_dataset(json_path=test_unknown_unknown_path,
                                                   transform=test_transform)
    test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))

    # When doing test, set the batch size to 1 to test the time one by one accurately
    test_known_known_loader = torch.utils.data.DataLoader(test_known_known_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              test_known_known_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)

    test_known_unknown_loader = torch.utils.data.DataLoader(test_known_unknown_dataset,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            sampler=torch.utils.data.RandomSampler(
                                                                test_known_unknown_index),
                                                            collate_fn=customized_dataloader.collate,
                                                            drop_last=True)

    test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                                              batch_size=1,
                                                              shuffle=False,
                                                              sampler=torch.utils.data.RandomSampler(
                                                                  test_unknown_unknown_index),
                                                              collate_fn=customized_dataloader.collate,
                                                              drop_last=True)

    ########################################################################
    # Create model: MSD-Net or other networks
    ########################################################################
    # TODO: change this to ResNet series
    if model_name == "resnet_50":
        model = torchvision.models.resnet50(pretrained=False)
        msd_net = False


    elif model_name == "resnet_101":
        model = torchvision.models.resnet101(pretrained=False)
        msd_net = False

    elif model_name == "resnet_152":
        model = torchvision.models.resnet152(pretrained=False)
        msd_net = False

    # MSD-Net here
    elif model_name == "msd_net":
        model = getattr(models, args.arch)(args)
        msd_net = True

    # TODO: Maybe adding other networks in the future (low priority)
    else:
        pass

    # Load trained model
    model.load_state_dict(torch.load(test_model_path))
    print("Loading model: %s" % test_model_path)

    # Generate features
    gen_feature(loader=train_known_known_loader,
                model=model,
                use_msd_net=msd_net,
                data_category="train_known_known",
                save_dir=model_path_base + model_dir)

    gen_feature(loader=train_known_unknown_loader,
                model=model,
                use_msd_net=msd_net,
                data_category="train_known_unknown",
                save_dir=model_path_base + model_dir)

    gen_feature(loader=test_known_known_loader,
                model=model,
                use_msd_net=msd_net,
                data_category="test_known_known",
                save_dir=model_path_base + model_dir)

    gen_feature(loader=test_known_unknown_loader,
                model=model,
                use_msd_net=msd_net,
                data_category="test_known_unknown",
                save_dir=model_path_base + model_dir)

    gen_feature(loader=test_unknown_unknown_loader,
                model=model,
                use_msd_net=msd_net,
                data_category="test_unknown_unknown",
                save_dir=model_path_base + model_dir)


if __name__ == '__main__':
    demo()