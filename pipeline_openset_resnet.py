import os
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
from utils import customized_dataloader
from utils.customized_dataloader import msd_net_dataset
from torchsummary import summary
import warnings
warnings.filterwarnings("ignore")
from args import arg_parser
import torch.nn as nn
import models
from datetime import datetime
from utils.pipeline_util import train_valid_test_one_epoch, \
    save_probs_and_features, find_best_model, update_thresholds, \
    save_openmax_feature, train_valid_test_one_epoch_for_resnet
from models import new_resnet

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

date = datetime.today().strftime('%Y-%m-%d')


###################################################################
                            # Loss options #
###################################################################
train_model = True

use_performance_loss = False
use_exit_loss = False

random_seed = 0

cross_entropy_weight = 1.0
perform_loss_weight = 1.0
exit_loss_weight = 1.0

unknown_ratio = None
unknown_weight = 0.0
update_threshold_freq = 5

###################################################################
                    # Training options #
###################################################################
run_msd_test = False
generate_openmax_feature = True


##########################
model_name = "msd_net"
debug = False
train_binary = False
use_addition = True


###################################################################
                    # Test process options #
###################################################################
"""
test_after_valid: testing MSD Net using the only last clf's prediction
run_test: testing with psyphy and exits
"""

# TODO: Cross-entropy seed 0 -- test_ce_00
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_0"

# TODO: Cross-entropy seed 1 -- test_ce_01
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_1"
#
# TODO: Cross-entropy seed 2 -- test_ce_02
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_2"

# TODO: Cross-entropy seed 3 -- test_ce_03
# test_model_dir = "2022-02-13/known_only_cross_entropy/seed_3"

# TODO: Cross-entropy seed 4 -- test_ce_04
test_model_dir = "2022-02-13/known_only_cross_entropy/seed_4"

#*******************************************************************#
# TODO: Cross-entropy + sam seed 0 -- test_sam_00
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_0"

# TODO: Cross-entropy + sam seed 1 -- test_sam_01
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_1"

# TODO: Cross-entropy + sam seed 2 -- test_sam_02
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_2"

# TODO: Cross-entropy + sam seed 3 -- test_sam_03
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_3"

# TODO: Cross-entropy + sam seed 4 -- test_sam_04
# test_model_dir = "2022-02-14/known_only_cross_entropy_1.0_pfm_1.0/seed_4"

#*******************************************************************#
# TODO: All 3 losses seed 0 -- test_all_loss_00
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_0"

# TODO: All 3 losses seed 1 -- test_all_loss_01
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_1"

# TODO: All 3 losses seed 2 -- test_all_loss_02
# test_model_dir = "2022-03-30/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_2"

# TODO: All 3 losses seed 3 -- test_all_loss_03
# test_model_dir = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_3"

# TODO: All 3 losses seed 4 -- test_all_loss_04
# test_model_dir = "2022-03-25/cross_entropy_1.0_pfm_1.0_exit_1.0_unknown_ratio_1.0/seed_4"


##################################################
use_trained_weights = True
run_one_sample = False
save_one_sample_rt_folder = None

# TODO: May need to change this in the future
test_model_base = "/afs/crc.nd.edu/user/j/jhuang24/Public/darpa_sail_on/models/msd_net"
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/models/openset_resnet"

test_model_path = test_model_base + "/" + test_model_dir
save_feat_path = save_path_base + "/" + test_model_dir

if test_model_path is not None:
    test_date = test_model_path.split("/")[-4]


###################################################################
                # Paths for saving results #
###################################################################
if (use_performance_loss == False) and (use_exit_loss == False):
    save_path_sub = "cross_entropy_only"

elif (use_performance_loss == True) and (use_exit_loss == False):
    save_path_sub = "cross_entropy_" + str(cross_entropy_weight) + \
                    "_pfm_" + str(perform_loss_weight)

elif (use_performance_loss == True) and (use_exit_loss == True):
    save_path_sub = "cross_entropy_" + str(cross_entropy_weight) + \
                    "_pfm_" + str(perform_loss_weight) + \
                    "_exit_" + str(exit_loss_weight)

elif (use_performance_loss == False) and (use_exit_loss == True):
    save_path_sub = "cross_entropy_" + str(cross_entropy_weight) + \
                    "_exit_" + str(exit_loss_weight)

else:
    save_path_sub = None

if unknown_ratio is not None:
    save_path_sub = save_path_sub + "_unknown_ratio_" + str(unknown_ratio)


####################################################################
    # Normally, there is no need to change these #
####################################################################
use_json_data = True
save_training_prob = False

nb_itr = 30
img_size = 224
lr = 0.1
wd = 0.0001
momentum=0.9

if debug:
    n_epochs = 3
else:
    n_epochs = 200

if run_msd_test:
    batch_size = 1
else:
    batch_size = 16


if train_binary:
    nb_training_classes = 2
else:
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
json_data_base_debug = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/" \
                       "data/object_recognition/image_net/derivatives/" \
                       "dataset_v1_3_partition/npy_json_files/2021_02_old"
json_data_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                 "dataset_v1_3_partition/npy_json_files_shuffled/"

if not run_msd_test:
    save_path_with_date = save_path_base + "/" + date
else:
    save_path_with_date = save_path_base

if not save_path_with_date:
    os.mkdir(save_path_with_date)

if debug:
    save_path = save_path_with_date + "/debug_" + save_path_sub + "/seed_" + str(random_seed)
else:
    save_path = save_path_with_date + "/" + save_path_sub + "/seed_" + str(random_seed)

if debug:
    train_known_known_path = os.path.join(json_data_base_debug, "debug_known_known.json")
    train_known_unknown_path = os.path.join(json_data_base_debug, "debug_known_unknown.json")

    valid_known_known_path = os.path.join(json_data_base_debug, "debug_known_known.json")
    valid_known_unknown_path = os.path.join(json_data_base_debug, "debug_known_unknown.json")

    test_known_known_path = os.path.join(json_data_base_debug, "debug_known_known.json")
    test_known_unknown_path = os.path.join(json_data_base_debug, "debug_known_unknown.json")
    test_unknown_unknown_path = os.path.join(json_data_base_debug, "debug_known_unknown.json")

else:
    train_known_known_path = os.path.join(json_data_base, "train_known_known.json")

    if unknown_ratio == 1.0:
        train_known_unknown_path = os.path.join(json_data_base, "train_known_unknown.json")
    else:
        train_known_unknown_path = json_data_base + "/train_known_unknown_" + str(unknown_ratio) + ".json"

    valid_known_known_path = os.path.join(json_data_base, "valid_known_known.json")
    valid_known_unknown_path = os.path.join(json_data_base, "valid_known_unknown.json")

    test_known_known_path = os.path.join(json_data_base, "test_known_known.json")

    test_known_known_path_p0 = os.path.join(json_data_base, "test_known_known_part_0.json")
    test_known_known_path_p1 = os.path.join(json_data_base, "test_known_known_part_1.json")
    test_known_known_path_p2 = os.path.join(json_data_base, "test_known_known_part_2.json")
    test_known_known_path_p3 = os.path.join(json_data_base, "test_known_known_part_3.json")

    test_known_unknown_path = os.path.join(json_data_base, "test_known_unknown.json")
    test_unknown_unknown_path = os.path.join(json_data_base, "test_unknown_unknown.json")



if __name__ == '__main__':
    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          normalize])

    valid_transform = train_transform

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])

    #######################################################################
    # Create dataset and data loader
    #######################################################################
    # Training
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

    # Validation
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

    # Test loaders
    test_known_known_dataset = msd_net_dataset(json_path=test_known_known_path,
                                                  transform=test_transform)
    test_known_known_index = torch.randperm(len(test_known_known_dataset))

    test_known_known_dataset_p0 = msd_net_dataset(json_path=test_known_known_path_p0,
                                               transform=test_transform)
    test_known_known_index_p0 = torch.randperm(len(test_known_known_dataset_p0))

    test_known_known_dataset_p1 = msd_net_dataset(json_path=test_known_known_path_p1,
                                                  transform=test_transform)
    test_known_known_index_p1 = torch.randperm(len(test_known_known_dataset_p1))

    test_known_known_dataset_p2 = msd_net_dataset(json_path=test_known_known_path_p2,
                                                  transform=test_transform)
    test_known_known_index_p2 = torch.randperm(len(test_known_known_dataset_p2))

    test_known_known_dataset_p3 = msd_net_dataset(json_path=test_known_known_path_p3,
                                                  transform=test_transform)
    test_known_known_index_p3 = torch.randperm(len(test_known_known_dataset_p3))


    test_known_unknown_dataset = msd_net_dataset(json_path=test_known_unknown_path,
                                                 transform=test_transform)
    test_known_unknown_index = torch.randperm(len(test_known_unknown_dataset))

    test_unknown_unknown_dataset = msd_net_dataset(json_path=test_unknown_unknown_path,
                                                   transform=test_transform)
    test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))

    # When doing test, set the batch size to 1 to test the time one by one accurately
    test_known_known_loader = torch.utils.data.DataLoader(test_known_known_dataset,
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         sampler=torch.utils.data.RandomSampler(
                                                             test_known_known_index),
                                                         collate_fn=customized_dataloader.collate,
                                                         drop_last=True)

    test_known_known_loader_p0 = torch.utils.data.DataLoader(test_known_known_dataset_p0,
                                                          batch_size=batch_size,
                                                          shuffle=False,
                                                          sampler=torch.utils.data.RandomSampler(
                                                              test_known_known_index_p0),
                                                          collate_fn=customized_dataloader.collate,
                                                          drop_last=True)

    test_known_known_loader_p1 = torch.utils.data.DataLoader(test_known_known_dataset_p1,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             sampler=torch.utils.data.RandomSampler(
                                                                 test_known_known_index_p1),
                                                             collate_fn=customized_dataloader.collate,
                                                             drop_last=True)

    test_known_known_loader_p2 = torch.utils.data.DataLoader(test_known_known_dataset_p2,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             sampler=torch.utils.data.RandomSampler(
                                                                 test_known_known_index_p2),
                                                             collate_fn=customized_dataloader.collate,
                                                             drop_last=True)

    test_known_known_loader_p3 = torch.utils.data.DataLoader(test_known_known_dataset_p3,
                                                             batch_size=batch_size,
                                                             shuffle=False,
                                                             sampler=torch.utils.data.RandomSampler(
                                                                 test_known_known_index_p3),
                                                             collate_fn=customized_dataloader.collate,
                                                             drop_last=True)

    test_known_unknown_loader = torch.utils.data.DataLoader(test_known_unknown_dataset,
                                                            batch_size=batch_size,
                                                            shuffle=False,
                                                            sampler=torch.utils.data.RandomSampler(
                                                                test_known_unknown_index),
                                                            collate_fn=customized_dataloader.collate,
                                                            drop_last=True)

    test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                                              batch_size=batch_size,
                                                              shuffle=False,
                                                              sampler=torch.utils.data.RandomSampler(
                                                                  test_unknown_unknown_index),
                                                              collate_fn=customized_dataloader.collate,
                                                              drop_last=True)

    ########################################################################
    # Create model: MSD-Net or other networks
    ########################################################################
    print("Creating resnet")
    model = new_resnet.resnet18()

    ########################################################################
    # Training + validation
    ########################################################################
    if train_model:
        # Make save directory
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.isdir(save_path):
            raise Exception('%s is not a dir' % save_path)

        # Setup random seed
        torch.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Model on cuda
        if torch.cuda.is_available():
            model = model.cuda()

        # Wrap model for multi-GPUs, if necessary
        model_wrapper = model

        # Optimizer
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model_wrapper.parameters(),
                                    lr=lr,
                                    momentum=momentum,
                                    nesterov=True,
                                    weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                         gamma=0.1)

        # Start log
        with open(os.path.join(save_path, 'results.csv'), 'w') as f:
            f.write('epoch, '
                    'train_loss, train_acc_top1, train_acc_top3, train_acc_top5, '
                    'train_exit_acc_top1, train_exit_acc_top3, train_exit_acc_top5,'
                    'valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5, '
                    'valid_exit_acc_top1, valid_exit_acc_top3, valid_exit_acc_top5\n')

        # Train model
        best_acc_top5 = 0.00

        for epoch in range(n_epochs):
            print("*" * 60)
            print("EPOCH:", epoch)

            train_loss, train_acc_top1,\
            train_acc_top3, train_acc_top5, \
            train_exit_acc_top1, train_exit_acc_top3, \
            train_exit_acc_top5 = train_valid_test_one_epoch_for_resnet(args=args,
                                                       known_loader=train_known_known_loader,
                                                       model=model_wrapper,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       nb_epoch=epoch,
                                                       use_msd_net=True,
                                                       train_phase=True,
                                                       save_path=save_path,
                                                       use_performance_loss=use_performance_loss,
                                                       use_exit_loss=use_exit_loss,
                                                       cross_entropy_weight=cross_entropy_weight,
                                                       perform_loss_weight=perform_loss_weight,
                                                       exit_loss_weight=exit_loss_weight,
                                                       known_exit_rt=known_exit_rt,
                                                       unknown_exit_rt=unknown_exit_rt,
                                                       known_thresholds=known_thresholds,
                                                       unknown_thresholds=unknown_thresholds,
                                                        unknown_loss_weight=unknown_weight)

            scheduler.step()

            valid_loss, valid_acc_top1, \
            valid_acc_top3, valid_acc_top5, \
            valid_exit_acc_top1, valid_exit_acc_top3, \
            valid_exit_acc_top5 = train_valid_test_one_epoch_for_resnet(args=args,
                                                       known_loader=valid_known_known_loader,
                                                       model=model_wrapper,
                                                       criterion=criterion,
                                                       optimizer=optimizer,
                                                       nb_epoch=epoch,
                                                       use_msd_net=True,
                                                       train_phase=False,
                                                       save_path=save_path,
                                                       use_performance_loss=use_performance_loss,
                                                       use_exit_loss=use_exit_loss,
                                                       cross_entropy_weight=cross_entropy_weight,
                                                       perform_loss_weight=perform_loss_weight,
                                                       exit_loss_weight=exit_loss_weight,
                                                       known_exit_rt=known_exit_rt,
                                                       unknown_exit_rt=unknown_exit_rt,
                                                       known_thresholds=known_thresholds,
                                                       unknown_thresholds=unknown_thresholds,
                                                       unknown_loss_weight=unknown_weight)



            if epoch % update_threshold_freq == 0:
                # TODO: Update known known thresholds for deciding exits
                print("Updating known known thresholds")
                updated_known_thresholds = update_thresholds(loader=valid_known_known_loader,
                                                      model=model_wrapper,
                                                      use_msd_net=True,
                                                      percentile=50)

                known_thresholds = updated_known_thresholds
                print("New known threshold: ", updated_known_thresholds)

            # Determine if model is the best
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
                print('New best top-5 validation accuracy: %.4f' % best_acc_top5)
                torch.save(model.state_dict(), save_path + "/model_epoch_" + str(epoch) + '.dat')
                torch.save(optimizer.state_dict(), save_path + "/optimizer_epoch_" + str(epoch) + '.dat')

            # Log results
            with open(os.path.join(save_path, 'results.csv'), 'a') as f:
                f.write('%03d, '
                        '%0.6f, %0.6f, %0.6f, %0.6f, '
                        '%0.6f, %0.6f, %0.6f,'
                        '%0.5f, %0.6f, %0.6f, %0.6f'
                        ' %0.6f, %0.6f, %0.6f \n'% ((epoch + 1),
                                                           train_loss, train_acc_top1, train_acc_top3, train_acc_top5,
                                                            train_exit_acc_top1, train_exit_acc_top3, train_exit_acc_top5,
                                                           valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5,
                                                            valid_exit_acc_top1, valid_exit_acc_top3, valid_exit_acc_top5))

    ########################################################################
    # Testing trained model
    ########################################################################
    if model_name == "msd_net":
        # TODO: find the best model in the given directory
        best_epoch, best_model_path = find_best_model(test_model_path)

        print("Best epoch:", best_epoch)
        print("Best model path:", best_model_path)

        model.load_state_dict(torch.load(best_model_path))
        print("Loading MSD-Net model: %s" % test_model_path)
        # print(model)

        if run_msd_test:
            # Create directories
            save_test_results_path = save_feat_path + "/test_results"
            if not os.path.exists(save_test_results_path):
                os.mkdir(save_test_results_path)

            save_all_feature_path = test_model_path + "/features"
            if not os.path.exists(save_all_feature_path):
                os.mkdir(save_all_feature_path)

            #################################################################
            # Run training and validation data
            #################################################################
            # Run process for testing and generating features
            print("Generating featrures and probabilities")

            save_probs_and_features(test_loader=valid_known_known_loader,
                                model=model,
                                test_type="valid_known_known",
                                use_msd_net=True,
                                epoch_index=best_epoch,
                                npy_save_dir=save_all_feature_path)

            save_probs_and_features(test_loader=valid_known_unknown_loader,
                                model=model,
                                test_type="valid_known_unknown",
                                use_msd_net=True,
                                epoch_index=best_epoch,
                                npy_save_dir=save_all_feature_path)

            ########################################################################
            # Testing data
            ########################################################################
            print("Testing models")
            print("Testing the known_known samples...")
            save_probs_and_features(test_loader=test_known_known_loader_p0,
                                model=model,
                                test_type="test_known_known",
                                use_msd_net=True,
                                epoch_index=best_epoch,
                                npy_save_dir=save_test_results_path,
                                part_index=0)

            save_probs_and_features(test_loader=test_known_known_loader_p1,
                                    model=model,
                                    test_type="test_known_known",
                                    use_msd_net=True,
                                    epoch_index=best_epoch,
                                    npy_save_dir=save_test_results_path,
                                    part_index=1)

            save_probs_and_features(test_loader=test_known_known_loader_p2,
                                    model=model,
                                    test_type="test_known_known",
                                    use_msd_net=True,
                                    epoch_index=best_epoch,
                                    npy_save_dir=save_test_results_path,
                                    part_index=2)

            save_probs_and_features(test_loader=test_known_known_loader_p3,
                                    model=model,
                                    test_type="test_known_known",
                                    use_msd_net=True,
                                    epoch_index=best_epoch,
                                    npy_save_dir=save_test_results_path,
                                    part_index=3)

            print("testing the unknown samples...")
            save_probs_and_features(test_loader=test_unknown_unknown_loader,
                                model=model,
                                test_type="unknown_unknown",
                                use_msd_net=True,
                                epoch_index=best_epoch,
                                npy_save_dir=save_test_results_path)

        elif generate_openmax_feature:
            # Train known known
            print("Train known known")
            save_openmax_feature(test_loader=train_known_known_loader,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/train_features")

            # Valid known known
            print("Valid known known")
            save_openmax_feature(test_loader=valid_known_known_loader,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/valid_features")

            # Test known known
            print("Test known known")
            save_openmax_feature(test_loader=test_known_known_loader_p0,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/test_known_features")

            save_openmax_feature(test_loader=test_known_known_loader_p1,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/test_known_features")

            save_openmax_feature(test_loader=test_known_known_loader_p2,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/test_known_features")

            save_openmax_feature(test_loader=test_known_known_loader_p3,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/test_known_features")

            # Test unknown unknown
            print("Test unknown unknown")
            save_openmax_feature(test_loader=test_unknown_unknown_loader,
                                 model=model,
                                 mat_save_dir=save_feat_path + "/openmax_feature/test_unknown_features")

        else:
            pass