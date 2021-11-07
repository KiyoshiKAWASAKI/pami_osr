import os
import time
import torch
import torchvision
from torchvision import datasets, transforms
from models import efficient_dense_net
import numpy as np
from timeit import default_timer as timer
from utils import customized_dataloader
from utils.customized_dataloader import msd_net_dataset, msd_net_with_grouped_rts
import sys
import warnings

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


###################################################################
# Training options #
###################################################################
model_name = "resnet_50"
save_path_sub = "resnet_50"

n_epochs = 200
batch_size = 16
nb_training_classes = 296
random_seed = 4

debug = False
run_test = False

####################################################################
# Normally, there is no need to change these #
####################################################################
use_json_data = True
save_training_prob = False

nb_itr = 30
nb_clfs = 5
img_size = 224
nBlocks = 5
nb_classes = 296
rt_max = 28


known_exit_rt = [3.5720, 4.9740, 7.0156, 11.6010, 27.5720]
unknown_exit_rt = [4.2550, 5.9220, 8.2368, 13.0090, 28.1661]

known_thresholds = [0.4692796697553254, 0.5056925101425871, 0.5137719005140328,
                    0.5123290032915468, 0.5468768758061252]
unknown_thresholds = [0.39571245688620443, 0.41746665012570583, 0.4149690186488925,
                      0.42355671497950664, 0.4600701578332428]

human_known_rt_max = 28
human_unknown_rt_max = 28
machine_known_rt_max = 0.057930
machine_unknown_rt_max = 0.071147

# Data from pp_add
train_known_known_machine_rt_max = [0.026294, 0.045157, 0.051007, 0.055542, 0.057930]
train_known_unknown_machine_rt_max = [0.032500, 0.064224, 0.067660, 0.070082, 0.071147]
# valid_known_known_machine_rt_max = [0.024027, 0.049423, 0.055970, 0.061030, 0.063622]
# valid_known_unknown_machine_rt_max = [0.010409, 0.015693, 0.019874, 0.022608, 0.023821]


#########################################################################################
# Define paths for saving model and data source #
#########################################################################################
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models/cvpr_resnet"
save_path_with_date = save_path_base + "/" + date

if not save_path_with_date:
    os.mkdir(save_path_with_date)

if debug:
    save_path = save_path_with_date + "/debug_" + save_path_sub + "_seed_" + str(random_seed)
else:
    save_path = save_path_with_date + "/" + save_path_sub + "_seed_" + str(random_seed)

if debug:
    train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                             "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
    train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_ne" \
                               "t/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
    valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                             "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
    valid_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                               "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"

    test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                            "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
    test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                              "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
    test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                                "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"


else:
    train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                             "dataset_v1_3_partition/npy_json_files/train_known_known.json"
    train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                               "dataset_v1_3_partition/npy_json_files/train_known_unknown.json"

    valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                             "dataset_v1_3_partition/npy_json_files/valid_known_known.json"
    valid_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                               "dataset_v1_3_partition/npy_json_files/valid_known_unknown.json"

    test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                            "dataset_v1_3_partition/npy_json_files/test_known_known_without_rt.json"
    test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                              "dataset_v1_3_partition/npy_json_files/test_known_unknown.json"
    test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/npy_json_files/test_unknown_unknown.json"


#########################################################################################
# Define all the functions #
#########################################################################################
def train_valid_one_epoch(known_loader,
                          unknown_loader,
                          model,
                          criterion,
                          optimizer,
                          nb_epoch,
                          train_phase,
                          nb_sample_per_bacth=16):
    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param penalty_factors_known:
    :param penalty_factors_unknown:
    :param use_msd_net:
    :param train_phase:
    :return:
    """

    ##########################################
    # Set up evaluation metrics
    ##########################################
    top1, top3, top5 = [], [], []

    losses = AverageMeter()
    top1.append(AverageMeter())
    top3.append(AverageMeter())
    top5.append(AverageMeter())

    if train_phase:
        model.train()
    else:
        model.eval()

    running_lr = None

    ###################################################
    # training process setup...
    ###################################################
    if train_phase:
        save_txt_path = os.path.join(save_path, "train_stats_epoch_" + str(nb_epoch) + ".txt")
    else:
        save_txt_path = os.path.join(save_path, "valid_stats_epoch_" + str(nb_epoch) + ".txt")

    # Count number of batches for known and unknown respectively
    nb_known_batches = len(known_loader)
    nb_unknown_batches = len(unknown_loader)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # Generate index for known and unknown and shuffle
    all_indices = random.sample(list(range(nb_total_batches)), len(list(range(nb_total_batches))))
    known_indices = all_indices[:nb_known_batches]
    unknown_indices = all_indices[nb_known_batches:]

    # Create iterator
    known_iter = iter(known_loader)
    unknown_iter = iter(unknown_loader)

    # Only train one batch for each step
    with open(save_txt_path, 'w') as f:
        for i in range(nb_total_batches):
            ##########################################
            # Basic setups
            ##########################################
            lr = adjust_learning_rate(optimizer, nb_epoch, args, batch=i,
                                      nBatch=nb_total_batches, method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            loss = 0.0

            ##########################################
            # Get a batch
            ##########################################
            if i in known_indices:
                batch = next(known_iter)

                input = batch["imgs"]
                target = batch["labels"]

            elif i in unknown_indices:
                batch = next(unknown_iter)

                input = batch["imgs"]
                target = batch["labels"]

                for i in range(len(target)):
                    target[i] = nb_training_classes - 1


            # Convert into PyTorch tensor
            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output, feature = model(input_var)

            # print(output)

            if not isinstance(output, list):
                output = [output]

            # print(len(output))
            # print(len(output[0][0]))

            for j in range(len(output)):
                # Cross-entropy loss only
                ce_loss = criterion(output[j], target_var)
                loss += ce_loss


            ##########################################
            # Calculate loss and BP
            ##########################################
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec3, prec5 = accuracy(output[j].data, target_var, topk=(1, 3, 5))
                top1[j].update(prec1.item(), input.size(0))
                top3[j].update(prec3.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            if train_phase:
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@3 {top3.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}\n'.format(
                    nb_epoch, i + 1, nb_total_batches,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                f.write('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f}\t'
                        'Acc@1 {top1.val:.4f}\t'
                        'Acc@3 {top3.val:.4f}\t'
                        'Acc@5 {top5.val:.4f}\n'.format(
                    nb_epoch, i + 1, nb_total_batches,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg


def train(model,
          train_known_known_loader,
          train_known_unknown_loader,
          valid_known_known_loader,
          valid_known_unknown_loader,
          save,
          lr=0.1,
          wd=0.0001,
          momentum=0.9,
          valid_loader=True,
          seed=None):
    """

    :param model:
    :param train_known_known_loader:
    :param train_known_unknown_loader:
    :param valid_known_known_loader:
    :param valid_known_unknown_loader:
    :param save:
    :param n_epochs:
    :param batch_size:
    :param lr:
    :param wd:
    :param momentum:
    :param seed:
    :return:
    """

    if seed is not None:
        torch.manual_seed(seed)

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    # if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    #     model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch, '
                'train_loss, train_acc_top1, train_acc_top3, train_acc_top5, '
                'valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5\n')

    # Train model
    best_acc_top1 = 0.00

    print("hello")

    for epoch in range(n_epochs):
        train_loss, train_acc_top1, \
        train_acc_top3, train_acc_top5 = train_valid_one_epoch(known_loader=train_known_known_loader,
                                                               unknown_loader=train_known_unknown_loader,
                                                               model=model_wrapper,
                                                               criterion=criterion,
                                                               optimizer=optimizer,
                                                               nb_epoch=epoch,
                                                               train_phase=True)

        scheduler.step()

        valid_loss, valid_acc_top1, \
        valid_acc_top3, valid_acc_top5 = train_valid_one_epoch(known_loader=valid_known_known_loader,
                                                               unknown_loader=valid_known_unknown_loader,
                                                               model=model_wrapper,
                                                               criterion=criterion,
                                                               optimizer=optimizer,
                                                               nb_epoch=epoch,
                                                               train_phase=False)


        # Determine if model is the best
        if valid_loader:
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                print('New best top-1 accuracy: %.4f' % best_acc_top1)
            torch.save(model.state_dict(), save + "/model_epoch_" + str(epoch) + '.dat')
            torch.save(optimizer.state_dict(), save + "/optimizer_epoch_" + str(epoch) + '.dat')
        else:
            torch.save(model.state_dict(), save + "/model_epoch_" + str(epoch) + '.dat')

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d, '
                    '%0.6f, %0.6f, %0.6f, %0.6f, '
                    '%0.5f, %0.6f, %0.6f, %0.6f,\n' % ((epoch + 1),
                                                       train_loss, train_acc_top1, train_acc_top3, train_acc_top5,
                                                       valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5))


def test_and_save_probs(test_loader,
                        model,
                        test_type,
                        use_msd_net,
                        epoch_index,
                        test_itr_index=None,
                        data_type=None):
    """
    batch size is always one for testing.

    :param test_loader:
    :param model:
    :param test_unknown:
    :param use_msd_net:
    :return:
    """
    # Setup the paths for saving npy files
    if use_trained_weights:
        if get_train_valid_prob:
            save_known_known_probs_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_known_prob_epoch_" + str(
                epoch_index) + ".npy"
            save_known_known_original_label_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_known_epoch_labels_epoch_" + str(
                epoch_index) + ".npy"
            save_known_known_rt_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_known_rts_epoch_" + str(
                epoch_index) + ".npy"

            save_known_unknown_probs_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_unknown_probs_epoch_" + str(
                epoch_index) + ".npy"  # save_known_unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/targets_epoch_" + str(epoch_index) + ".npy"
            save_known_unknown_original_label_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_unknown_labels_epoch_" + str(
                epoch_index) + ".npy"
            save_known_unknown_rt_path = save_path_with_date + "/" + save_path_sub + "/test/" + data_type + "_known_unknown_rts_epoch_" + str(
                epoch_index) + ".npy"

        else:
            save_prob_base_path = save_path_with_date + "/" + save_path_sub + "/test"

            if not os.path.exists(save_prob_base_path):
                print("Creating the directory %s" % save_prob_base_path)
                os.mkdir(save_prob_base_path)

    else:
        # Test the model with random initialized weights
        # Test: train_known_known, train_known_unknown
        # save_known_known_probs_path = save_path_base + "/" + save_path_sub + "/test/known_known_prob_rand_" + str(test_itr_index) + ".npy"
        # save_known_known_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_known_labels_rand_" + str(test_itr_index) + ".npy"
        # save_known_known_rt_path = save_path_base + "/" + save_path_sub + "/test/known_known_rts_rand_" + str(test_itr_index) + ".npy"
        #
        # save_known_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_probs_rand_" + str(test_itr_index) + ".npy"  # save_known_unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/targets_epoch_" + str(epoch_index) + ".npy"
        # save_known_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_labels_rand_" + str(test_itr_index) + ".npy"
        # save_known_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/known_unknown_rts_rand_" + str(test_itr_index) + ".npy"
        pass

    # Set the model to evaluation mode
    model.cuda()
    model.eval()

    # Define the softmax - do softmax to each block.
    if use_msd_net:
        print("Testing MSD-Net...")
        sm = torch.nn.Softmax(dim=2)

        # For MSD-Net, save everything into npy files
        full_original_label_list = []
        full_prob_list = []
        full_rt_list = []

        print(len(test_loader))
        # sys.exit()

        for i in range(len(test_loader)):
            # for i in range(5):
            batch = next(iter(test_loader))

            input = batch["imgs"]
            target = batch["labels"] - 1

            rts = []
            input = input.cuda()
            target = target.cuda(async=True)

            # Save original labels to the list
            original_label_list = np.array(target.cpu().tolist())
            for label in original_label_list:
                full_original_label_list.append(label)

            input_var = torch.autograd.Variable(input)

            # Get the model outputs and RTs
            start = timer()
            output, end_time = model(input_var)

            # print(end_time)

            # Save the RTs
            # TODO: something is diff in RT - 0327
            for end in end_time[0]:
                print("Processes one sample in %f sec" % (end - start))
                rts.append(end - start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            prob = sm(torch.stack(output).to())  # Shape is [block, batch, class]
            prob_list = np.array(prob.cpu().tolist())

            # Reshape it into [batch, block, class]
            prob_list = np.reshape(prob_list,
                                   (prob_list.shape[1],
                                    prob_list.shape[0],
                                    prob_list.shape[2]))

            for one_prob in prob_list.tolist():
                full_prob_list.append(one_prob)

        # Save all results to npy
        full_original_label_list_np = np.array(full_original_label_list)
        full_prob_list_np = np.array(full_prob_list)
        full_rt_list_np = np.array(full_rt_list)

        if use_trained_weights:
            if get_train_valid_prob:
                if test_type == "known_known":
                    print("Saving probabilities to %s" % save_known_known_probs_path)
                    np.save(save_known_known_probs_path, full_prob_list_np)
                    print("Saving original labels to %s" % save_known_known_original_label_path)
                    np.save(save_known_known_original_label_path, full_original_label_list_np)
                    print("Saving RTs to %s" % save_known_known_rt_path)
                    np.save(save_known_known_rt_path, full_rt_list_np)

                elif test_type == "known_unknown":
                    print("Saving probabilities to %s" % save_known_unknown_probs_path)
                    np.save(save_known_unknown_probs_path, full_prob_list_np)
                    print("Saving original labels to %s" % save_known_unknown_original_label_path)
                    np.save(save_known_unknown_original_label_path, full_original_label_list_np)
                    print("Saving RTs to %s" % save_known_unknown_rt_path)
                    np.save(save_known_unknown_rt_path, full_rt_list_np)

            else:
                if test_type == "known_known":
                    save_known_known_probs_path = save_prob_base_path + "/known_known_probs_epoch_" + str(
                        epoch_index) + ".npy"
                    save_known_known_original_label_path = save_prob_base_path + "/known_known_labels_epoch_" + str(
                        epoch_index) + ".npy"
                    save_known_known_rt_path = save_prob_base_path + "/known_known_rts_epoch_" + str(
                        epoch_index) + ".npy"

                    print("Saving probabilities to %s" % save_known_known_probs_path)
                    np.save(save_known_known_probs_path, full_prob_list_np)
                    print("Saving original labels to %s" % save_known_known_original_label_path)
                    np.save(save_known_known_original_label_path, full_original_label_list_np)
                    print("Saving RTs to %s" % save_known_known_rt_path)
                    np.save(save_known_known_rt_path, full_rt_list_np)

                elif test_type == "known_unknown":
                    save_known_unknown_probs_path = save_prob_base_path + "/known_unknown_probs_epoch_" + str(
                        epoch_index) + ".npy"
                    save_known_unknown_original_label_path = save_prob_base_path + "/known_unknown_labels_epoch_" + str(
                        epoch_index) + ".npy"
                    save_known_unknown_rt_path = save_prob_base_path + "/known_unknown_rts_epoch_" + str(
                        epoch_index) + ".npy"

                    print("Saving probabilities to %s" % save_known_unknown_probs_path)
                    np.save(save_known_unknown_probs_path, full_prob_list_np)
                    print("Saving original labels to %s" % save_known_unknown_original_label_path)
                    np.save(save_known_unknown_original_label_path, full_original_label_list_np)
                    print("Saving RTs to %s" % save_known_unknown_rt_path)
                    np.save(save_known_unknown_rt_path, full_rt_list_np)

                else:
                    save_unknown_unknown_probs_path = save_prob_base_path + "/unknown_unknown_probs_epoch_" + str(
                        epoch_index) + ".npy"
                    save_unknown_unknown_original_label_path = save_prob_base_path + "/unknown_unknown_labels_epoch_" + str(
                        epoch_index) + ".npy"
                    save_unknown_unknown_rt_path = save_prob_base_path + "/unknown_unknown_rts_epoch_" + str(
                        epoch_index) + ".npy"

                    print("Saving probabilities to %s" % save_unknown_unknown_probs_path)
                    np.save(save_unknown_unknown_probs_path, full_prob_list_np)
                    print("Saving original labels to %s" % save_unknown_unknown_original_label_path)
                    np.save(save_unknown_unknown_original_label_path, full_original_label_list_np)
                    print("Saving RTs to %s" % save_unknown_unknown_rt_path)
                    np.save(save_unknown_unknown_rt_path, full_rt_list_np)

        else:
            if test_type == "known_known":
                print("Saving probabilities to %s" % save_known_known_probs_path)
                np.save(save_known_known_probs_path, full_prob_list_np)
                print("Saving original labels to %s" % save_known_known_original_label_path)
                np.save(save_known_known_original_label_path, full_original_label_list_np)
                print("Saving RTs to %s" % save_known_known_rt_path)
                np.save(save_known_known_rt_path, full_rt_list_np)

            elif test_type == "known_unknown":
                print("Saving probabilities to %s" % save_known_unknown_probs_path)
                np.save(save_known_unknown_probs_path, full_prob_list_np)
                print("Saving original labels to %s" % save_known_unknown_original_label_path)
                np.save(save_known_unknown_original_label_path, full_original_label_list_np)
                print("Saving RTs to %s" % save_known_unknown_rt_path)
                np.save(save_known_unknown_rt_path, full_rt_list_np)



    # TODO: Test process for other networks - is it different??
    else:
        pass


def run_one_sample(test_loader,
                   model,
                   use_msd_net,
                   save_folder,
                   test_itr_index):
    """
    Run one sample thru different models n times and
    see whether there is a pattern for RTs.

    :param test_loader:
    :param model:
    :param test_unknown:
    :param use_msd_net:
    :return:
    """

    # Set the model to evaluation mode
    model.cuda()
    model.eval()

    # Define the softmax - do softmax to each block.
    if use_msd_net:
        print("Testing MSD-Net...")
        sm = torch.nn.Softmax(dim=2)

        # For MSD-Net, save everything into npy files
        full_original_label_list = []
        full_prob_list = []
        full_rt_list = []
        print(len(test_loader))

        # Only process one image
        for i in range(int(round(len(test_loader) / 3))):
            batch = next(iter(test_loader))

            input = batch["imgs"]
            target = batch["labels"] - 1

            rts = []
            input = input.cuda()
            target = torch.tensor(target).cuda(async=True)

            # Save original labels to the list
            original_label_list = np.array(target.cpu().tolist())
            for label in original_label_list:
                full_original_label_list.append(label)

            input_var = torch.autograd.Variable(input)

            # Get the model outputs and RTs
            start = timer()
            output, end_time = model(input_var)

            # print(end_time)

            # Save the RTs
            # TODO: something is diff in RT - 0327
            for end in end_time[0]:
                print("Processes one sample in %f sec" % (end - start))
                rts.append(end - start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            prob = sm(torch.stack(output).to())  # Shape is [block, batch, class]
            prob_list = np.array(prob.cpu().tolist())

            # Reshape it into [batch, block, class]
            prob_list = np.reshape(prob_list,
                                   (prob_list.shape[1],
                                    prob_list.shape[0],
                                    prob_list.shape[2]))

            for one_prob in prob_list.tolist():
                full_prob_list.append(one_prob)

        # Save all results to npy
        full_original_label_list_np = np.array(full_original_label_list)
        full_prob_list_np = np.array(full_prob_list)
        full_rt_list_np = np.array(full_rt_list)

        save_rt_path = save_folder + "/rt_itr_" + str(test_itr_index) + ".npy"
        print("Saving RTs to %s" % save_rt_path)
        np.save(save_rt_path, full_rt_list_np)

    # TODO: Test process for other networks - is it different??
    else:
        pass


def demo():
    global args

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
    # Create resnet model here
    if model_name == "resnet_50":
        model = torchvision.models.resnet50(pretrained=False)

    elif model_name == "resnet_101":
        model = torchvision.models.resnet101(pretrained=False)

    elif model_name == "resnet_152":
        model = torchvision.models.resnet152(pretrained=False)

    else:
        model = None


    # Make save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)

    # Combine training all networks together
    train(model=model,
          train_known_known_loader=train_known_known_loader,
          train_known_unknown_loader=train_known_unknown_loader,
          valid_known_known_loader=valid_known_known_loader,
          valid_known_unknown_loader=valid_known_unknown_loader,
          save=save_path)


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """

    :param output:
    :param target:
    :param topk:
    :return:
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    demo()