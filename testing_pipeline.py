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
from utils.pipeline_util import train_valid_test_one_epoch, test_and_save_probs

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

date = datetime.today().strftime('%Y-%m-%d')


"""
Testing pipeline:

1. Just provide the folder for the all the models
2. Search the best model in the given folder
3. Generate a bunch of stuff
    3.1 Generate probs from validation data - get thresholds for this model
    3.2 Testing model and generate probs
    3.3 Generate features for EVM and SVM
"""



test_folder_sub_dir = ""

###################################################################################
# Data paths (usually, no need to change these)
###################################################################################
depth=100
growth_rate=12
efficient=True
unknown_ratio = 1.0

test_folder_base_dir = "/scratch365/jhuang24/sail-on/models/msd_net/"

json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

train_known_known_path = os.path.join(json_data_base, "train_known_known.json")

if unknown_ratio is not None:
    if unknown_ratio == 1.0:
        train_known_unknown_path = os.path.join(json_data_base, "train_known_unknown.json")
    else:
        train_known_unknown_path = json_data_base + "/train_known_unknown_" + str(unknown_ratio) + ".json"

valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")
valid_known_unknown_path = os.path.join(json_data_base, "valid_known_unknown.json")

test_known_known_path = os.path.join(json_data_base, "test_known_known.json")
test_known_unknown_path = os.path.join(json_data_base, "test_known_unknown.json")
test_unknown_unknown_path = os.path.join(json_data_base, "test_unknown_unknown.json")





###################################################################################
# Functions
###################################################################################
def find_best_model(dir_to_models):
    """

    :param dir_to_models:
    :return:
    """


    return





if __name__ == '__main__':
    #####################################################
    # Generate model
    #####################################################
    # Get densenet configuration
    # global args

    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = getattr(models, args.arch)(args)



