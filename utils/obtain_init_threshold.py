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
from utils.pipeline_util import save_probs_and_features

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

date = datetime.today().strftime('%Y-%m-%d')


###################################################################
                        #  #
###################################################################
model_name = "msd_net"
img_size = 224
lr = 0.1
wd = 0.0001
momentum = 0.9
batch_size = 16
nb_training_classes = 294


###################################################################
save_path_base = "/scratch365/jhuang24/sail-on/"
save_all_feature_path = save_path_base + "/thresh_feat"


###################################################################
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")
valid_known_unknown_path = os.path.join(json_data_base, "valid_known_unknown.json")


if __name__ == '__main__':
    depth=100
    growth_rate=12
    efficient=True

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    valid_transform = transforms.Compose([transforms.RandomResizedCrop(224),
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
    valid_known_known_dataset = msd_net_dataset(json_path=valid_known_known_path,
                                                         transform=valid_transform)
    valid_known_unknown_dataset = msd_net_dataset(json_path=valid_known_unknown_path,
                                                           transform=valid_transform)

    valid_known_known_index = torch.randperm(len(valid_known_known_dataset))
    valid_known_unknown_index = torch.randperm(len(valid_known_unknown_dataset))

    valid_known_known_loader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                                           batch_size=batch_size,
                                                           shuffle=False,
                                                           collate_fn=customized_dataloader.collate,
                                                           sampler=torch.utils.data.RandomSampler(
                                                               valid_known_known_index))
    valid_known_unknown_loader = torch.utils.data.DataLoader(valid_known_unknown_dataset,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             drop_last=True,
                                                             collate_fn=customized_dataloader.collate,
                                                             sampler=torch.utils.data.RandomSampler(
                                                                 valid_known_unknown_index))


    ########################################################################
    # Create model: MSD-Net or other networks
    ########################################################################
    if model_name == "dense_net":
        print("Creating DenseNet")
        model = efficient_dense_net.DenseNet(growth_rate=growth_rate,
                                            block_config=block_config,
                                            num_init_features=growth_rate * 2,
                                            num_classes=nb_training_classes,
                                            small_inputs=True,
                                            efficient=efficient)

    # Add creating MSD-Net here
    elif model_name == "msd_net":
        model = getattr(models, args.arch)(args)

    # TODO (low priority): Maybe adding other networks in the future
    else:
        model = None

    #################################################################
    # Run training and validation data
    #################################################################
    # TODO: Run process for testing and generating features
    print("Generating featrures and probabilities")
    save_probs_and_features(test_loader=valid_known_known_loader,
                            model=model,
                            test_type="valid_known_known",
                            use_msd_net=True,
                            epoch_index=0,
                            npy_save_dir=save_all_feature_path)

    save_probs_and_features(test_loader=valid_known_unknown_loader,
                            model=model,
                            test_type="valid_known_unknown",
                            use_msd_net=True,
                            epoch_index=0,
                            npy_save_dir=save_all_feature_path)


