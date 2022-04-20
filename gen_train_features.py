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
                    # Testing model paths #
###################################################################
# TODO: Cross-entropy seed 0 -- feat_00
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_0"
# epoch = 147

# TODO: Cross-entropy seed 1 -- feat_01
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_1"
# epoch = 181

# TODO: Cross-entropy seed 2 -- feat_02
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_2"
# epoch = 195

# TODO: Cross-entropy seed 3 -- feat_03
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_3"
# epoch = 142

# TODO: Cross-entropy seed 4 -- feat_04
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_4"
# epoch = 120

#*******************************************************************#
# # TODO: Cross-entropy + sam seed 0 -- feat_05
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0"
# epoch = 175

# TODO: Cross-entropy + sam seed 1 -- feat_06
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_1"
# epoch = 105

# TODO: Cross-entropy + sam seed 2 -- feat_07
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_2"
# epoch = 159

# TODO: Cross-entropy + sam seed 3 -- feat_08
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_3"
# epoch = 103

# TODO: Cross-entropy + sam seed 4 -- feat_09
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_4"
# epoch = 193

#*******************************************************************#
# TODO: All 3 losses seed 0 -- feat_10
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0"
# epoch = 156

# TODO: All 3 losses seed 1 -- feat_11
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_1"
# epoch = 194

# TODO: All 3 losses seed 2 -- feat_12
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_2"
# epoch = 192

# TODO: All 3 losses seed 3 -- feat_13
# test_model_dir = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_3"
# epoch = 141

# TODO: All 3 losses seed 4 -- feat_14
# test_model_dir = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_4"
# epoch = 160

#*******************************************************************#
# TODO: CE + pp seed 0 -- feat_15
# test_model_dir = "2022-03-25/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_0"
# epoch = 173

# TODO: CE + pp seed 1 -- feat_16
# test_model_dir = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_1"
# epoch = 130

# TODO: CE + pp seed 2 -- feat_17
test_model_dir = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_2"
epoch = 166

# TODO: CE + pp seed 3 -- feat_18
# test_model_dir = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_3"
# epoch = 128

# TODO: CE + pp seed 4 -- feat_19
# test_model_dir = "2022-03-30/cross_entropy_1.0_exit_1.0_unknown_ratio_1.0/seed_4"
# epoch = 110

##################################################
batch_size = 16

model_path_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
save_feature_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/msd_net"
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

test_model_path = model_path_base + "/" + test_model_dir
save_feature_dir = os.path.join(save_feature_base, test_model_dir)
test_unknown_unknown_path = os.path.join(json_data_base, "test_unknown_unknown.json")


if __name__ == '__main__':

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    test_unknown_unknown_dataset = msd_net_dataset(json_path=test_unknown_unknown_path,
                                                transform=test_transform)
    test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))
    test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           drop_last=True,
                                                           collate_fn=customized_dataloader.collate,
                                                           sampler=torch.utils.data.RandomSampler(
                                                               test_unknown_unknown_index))

    # Create model: MSD-Net or other networks
    model = getattr(models, args.arch)(args)

    # Testing trained model
    test_model = test_model_path + "/model_epoch_" + str(epoch) + ".dat"
    model.load_state_dict(torch.load(test_model))
    print("Loading MSD-Net model: %s" % test_model_path)

    # Create directories
    if not os.path.exists(save_feature_dir):
        os.mkdir(save_feature_dir)


    #################################################################
    # Run training and validation data
    #################################################################
    # Run process for testing and generating features
    print("Generating featrures and probabilities")
    save_probs_and_features(test_loader=test_unknown_unknown_loader,
                        model=model,
                        test_type="test_unknown_unknown",
                        use_msd_net=True,
                        epoch_index=epoch,
                        npy_save_dir=save_feature_dir)

