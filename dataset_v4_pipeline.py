# Test pipeline only for UCCS dataset v4

import os
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
from utils import customized_dataloader
from utils.customized_dataloader import msd_net_dataset
import warnings
warnings.filterwarnings("ignore")
from args import arg_parser
import torch.nn as nn
import models
from datetime import datetime
from utils.pipeline_util import train_valid_test_one_epoch, \
    save_probs_and_features, find_best_model, update_thresholds

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

date = datetime.today().strftime('%Y-%m-%d')


###################################################################
                    # All path #
###################################################################
# TODO: Cross-entropy seed 0 -- test_ce_00
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_0/model_epoch_147.dat"
# model = "cross_entropy_only"
# best_epoch = 147

# TODO: Cross-entropy + sam seed 0 -- test_sam_00
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0/model_epoch_175.dat"
# model = "cross_entropy_pfm"
# best_epoch = 175

# TODO: All 3 losses seed 0 -- test_all_loss_00
test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0/model_epoch_156.dat"
model = "psyphy"
best_epoch = 156

##################################################################
# TODO: May need to change this in the future
test_model_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models"

test_model_path = test_model_base + "/" + test_model_dir
save_result_path = save_path_base + "/test_uccs/" + model

if not os.path.exists(save_result_path):
    os.mkdir(save_result_path)
    print("New dir created: ", save_result_path)

nb_training_classes = 293
known_exit_rt = [3.5720, 4.9740, 7.0156, 11.6010, 27.5720]
unknown_exit_rt = [4.2550, 5.9220, 8.2368, 13.0090, 28.1661]

# TODO: Need to update these thresholds
known_thresholds = [0.0035834426525980234, 0.0035834424197673798,
                    0.0035834426525980234, 0.0035834424197673798, 0.0035834424197673798]
unknown_thresholds = [0.0035834426525980234, 0.0035834424197673798,
                      0.0035834426525980234, 0.0035834424197673798, 0.0035834424197673798]


#########################################################################################
            # Define paths for saving model and data source #
#########################################################################################
# Normally, no need to change these
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/dataset_v4/json_files"

known_subject_07 = os.path.join(json_data_base, "known_subject_07.json")
unknown_subject_07 = os.path.join(json_data_base, "unknown_subject_07.json")
known_object_07 = os.path.join(json_data_base, "known_object_07.json")
unknown_object_07 = os.path.join(json_data_base, "unknown_object_07.json")

known_subject_08 = os.path.join(json_data_base, "known_subject_08.json")
unknown_subject_08 = os.path.join(json_data_base, "unknown_subject_08.json")
known_object_08 = os.path.join(json_data_base, "known_object_08.json")
unknown_object_08 = os.path.join(json_data_base, "unknown_object_08.json")


if __name__ == '__main__':
    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    # Test 07
    known_subject_07_dataset = msd_net_dataset(json_path=known_subject_07,
                                                  transform=test_transform)
    known_subject_07_index = torch.randperm(len(known_subject_07_dataset))

    known_object_07_dataset = msd_net_dataset(json_path=known_object_07,
                                               transform=test_transform)
    known_object_07_index = torch.randperm(len(known_object_07_dataset))

    unknown_subject_07_dataset = msd_net_dataset(json_path=unknown_subject_07,
                                               transform=test_transform)
    unknown_subject_07_index = torch.randperm(len(unknown_subject_07_dataset))

    unknown_object_07_dataset = msd_net_dataset(json_path=unknown_object_07,
                                              transform=test_transform)
    unknown_object_07_index = torch.randperm(len(unknown_object_07_dataset))

    # Test 08
    known_subject_08_dataset = msd_net_dataset(json_path=known_subject_08,
                                               transform=test_transform)
    known_subject_08_index = torch.randperm(len(known_subject_08_dataset))

    known_object_08_dataset = msd_net_dataset(json_path=known_object_08,
                                              transform=test_transform)
    known_object_08_index = torch.randperm(len(known_object_08_dataset))

    unknown_subject_08_dataset = msd_net_dataset(json_path=unknown_subject_08,
                                                 transform=test_transform)
    unknown_subject_08_index = torch.randperm(len(unknown_subject_08_dataset))

    unknown_object_08_dataset = msd_net_dataset(json_path=unknown_object_08,
                                                transform=test_transform)
    unknown_object_08_index = torch.randperm(len(unknown_object_08_dataset))

    # Test 07
    known_subject_07_loader = torch.utils.data.DataLoader(known_subject_07_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             known_subject_07_index),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

    unknown_subject_07_loader = torch.utils.data.DataLoader(unknown_subject_07_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              unknown_subject_07_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)

    known_object_07_loader = torch.utils.data.DataLoader(known_object_07_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              known_object_07_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)

    unknown_object_07_loader = torch.utils.data.DataLoader(unknown_object_07_dataset,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            sampler=torch.utils.data.RandomSampler(
                                                                unknown_object_07_index),
                                                            collate_fn=customized_dataloader.collate,
                                                            drop_last=True)

    # Test 08
    known_subject_08_loader = torch.utils.data.DataLoader(known_subject_08_dataset,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              known_subject_08_index),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)

    unknown_subject_08_loader = torch.utils.data.DataLoader(unknown_subject_08_dataset,
                                                            batch_size=1,
                                                            shuffle=False,
                                                            sampler=torch.utils.data.RandomSampler(
                                                                unknown_subject_08_index),
                                                            collate_fn=customized_dataloader.collate,
                                                            drop_last=True)

    known_object_08_loader = torch.utils.data.DataLoader(known_object_08_dataset,
                                                         batch_size=1,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             known_object_08_index),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

    unknown_object_08_loader = torch.utils.data.DataLoader(unknown_object_08_dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           sampler=torch.utils.data.RandomSampler(
                                                               unknown_object_08_index),
                                                           collate_fn=customized_dataloader.collate,
                                                           drop_last=True)

    ########################################################################
    # Load model and test
    ########################################################################
    model = getattr(models, args.arch)(args)
    model.load_state_dict(torch.load(test_model_path))
    print("Loading MSD-Net model: %s" % test_model_path)

    # Testing 07
    save_probs_and_features(test_loader=known_subject_07_loader,
                            model=model,
                            test_type="known_subject_07",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=unknown_subject_07_loader,
                            model=model,
                            test_type="unknown_subject_07",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=known_object_07_loader,
                            model=model,
                            test_type="known_object_07",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=unknown_object_07_loader,
                            model=model,
                            test_type="unknown_object_07",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    # Test 08
    save_probs_and_features(test_loader=known_subject_08_loader,
                            model=model,
                            test_type="known_subject_08",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=unknown_subject_08_loader,
                            model=model,
                            test_type="unknown_subject_08",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=known_object_08_loader,
                            model=model,
                            test_type="known_object_08",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)

    save_probs_and_features(test_loader=unknown_object_08_loader,
                            model=model,
                            test_type="unknown_object_08",
                            use_msd_net=True,
                            epoch_index=best_epoch,
                            npy_save_dir=save_result_path)
