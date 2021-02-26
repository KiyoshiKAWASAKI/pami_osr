import fire
import os
import time
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
import numpy as np
from timeit import default_timer as timer
from utils import customized_dataloader
from utils.customized_dataloader import msd_net_dataset
from op_counter import measure_model
import sys
import warnings
warnings.filterwarnings("ignore")
import random
from args import arg_parser
import torch.nn as nn
import models
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

# logging.basicConfig(filename=log_file_path,
#                     level=logging.INFO,
#                     format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')

# logging.basicConfig(stream=sys.stdout,
#                     level=logging.INFO,
#                     format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')
#
#
# log_file_path = args.log_file_path

# Define the TensorBoard
# writer = SummaryWriter()
# writer = SummaryWriter(args.tf_board_path)


###############################################
# Change these parameters
###############################################
model_name = "msd_net"
# model_name = "dense_net"
# model_name = "inception_v4"
# model_name = "vgg16"
debug = False
use_pp_loss = True

use_pre_train = False
train_binary = False
run_test = False


# This is for the binary classifier
test_msd_base_epoch = [0]
test_msd_5_weights_epoch = [0]
test_msd_pp_epoch = [0]


# This is the path for loading and testing model
# model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/" \
#              "combo_pipeline/1203/msd_5_weights_pp/model_epoch_14.dat"

# This is for saving training model as well as saving test npys
save_path_sub = "0225/pp_loss"

# This is the path for the pre-train model used for continue training
pre_train_model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/" \
                       "sail-on/combo_pipeline/1214_addition_full_set/msd_base/model_epoch_99.dat"


###############################################
# Noromaly, there is no need to change these
###############################################
use_json_data = True
save_training_prob = False

n_epochs = 200
batch_size = 16

img_size = 224
nBlocks = 5
thresh_top_1 = 0.90

if debug:
    nb_classes = 336
else:
    nb_classes = 296

if train_binary:
    nb_training_classes = 2
else:
    if debug:
        nb_training_classes = 336  # known_known:335, unknown_unknown:1
    else:
        nb_training_classes = 296 # known_known:295, unknown_unknown:1



save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/models"

if debug:
    train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                             "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
    train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_ne" \
                               "t/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
    valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
                             "/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
    valid_known_unknown_path =  "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net" \
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
    valid_known_unknown_path =  "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/npy_json_files/valid_known_unknown.json"

    test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                            "dataset_v1_3_partition/npy_json_files/test_known_known_without_rt.json"
    test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                              "dataset_v1_3_partition/npy_json_files/test_known_unknown.json"
    test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/npy_json_files/test_unknown_unknown.json"

save_path = save_path_base + "/" + save_path_sub




def train_valid_one_epoch(known_loader,
                            unknown_loader,
                            model,
                            criterion,
                            optimizer,
                            nb_epoch,
                            use_msd_net,
                            train_phase):
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
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1, top3, top5 = [], [], []

    if use_msd_net:
        for i in range(nBlocks):
            top1.append(AverageMeter())
            top3.append(AverageMeter())
            top5.append(AverageMeter())
    else:
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    if train_phase:
        model.train()
    else:
        model.eval()

    end = time.time()

    running_lr = None

    ###################################################
    # training process setup...
    ###################################################
    if train_phase:
        save_txt_path = os.path.join(save_path, "train_stats_epoch_" + str(nb_epoch) + ".txt")
        # save_prob_path = os.path.join(save_path, "train_probs_epoch_" + str(nb_epoch) + ".npy")
        # save_label_path =
    else:
        save_txt_path = os.path.join(save_path, "valid_stats_epoch_" + str(nb_epoch) + ".txt")
        # save_prob_path = os.path.join(save_path, "valid_probs_epoch_" + str(nb_epoch) + ".npy")

    # Count number of batches for known and unknown respectively
    nb_known_batches = len(known_loader)
    nb_unknown_batches = len(unknown_loader)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # TODO: Add this part for tensorboard
    # Check which category has more samples/batches
    if nb_known_batches >= nb_unknown_batches:
        base_step = len(known_loader)
    else:
        base_step = len(unknown_loader)

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

            data_time.update(time.time() - end)

            loss = 0.0

            ##########################################
            # Get a batch
            ##########################################
            if i in known_indices:
                batch = next(known_iter)
                # batch_type = "known"

            elif i in unknown_indices:
                batch = next(unknown_iter)
                # batch_type = "unknown"

            input = batch["imgs"]
            rts = batch["rts"]
            target = batch["labels"]

            # Change label into binary
            if train_binary:
                for i in range(len(target)):
                    one_target = target[i]

                    if one_target < nb_classes:
                        target[i] = 0
                    else:
                        target[i] = 1

            # print(target)

            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            # print(type(output))

            ##########################################
            # Only MSD-Net
            ##########################################
            if model_name == "msd_net":
                # Case 1: Not using pp loss
                if use_pp_loss == False:
                    for j in range(len(output)):
                        loss += criterion(output[j], target_var)

                # Case 2: Using pp loss
                if use_pp_loss == True:
                    print("Using psyphy loss")
                    for j in range(len(output)):
                        scale_factor = get_pp_factor(rts[j])
                        loss += scale_factor * criterion(output[j], target_var)

            else:
                # TODO: other networks - may be the same with MSD Net cause 5 weights are gone?
                pass


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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                # TODO: Implement TensorBoard
                # writer.add_scalar('training loss', losses.val, i * nb_epoch + base_step)
                # writer.add_scalar('Acc top-1', top1[-1].val, i * nb_epoch + base_step)
                # writer.add_scalar('Acc top-3', top3[-1].val, i * nb_epoch + base_step)
                # writer.add_scalar('Acc top-5', top5[-1].val, i * nb_epoch + base_step)

                print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    nb_epoch, i + 1, nb_total_batches,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    nb_epoch, i + 1, nb_total_batches,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

        # TODO: Get probability and save after each epoch (??)





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

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    if use_pre_train:
        print("Keep training on a pre-train 100 epoch model")
        checkpoint = torch.load(pre_train_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch, '
                'train_loss, train_acc_top1, train_acc_top3, train_acc_top5, '
                'valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5\n')

    # Train model
    best_acc_top1 = 0.00

    for epoch in range(n_epochs):
        if model_name == "msd_net":
            train_loss, train_acc_top1, \
            train_acc_top3, train_acc_top5 = train_valid_one_epoch(known_loader=train_known_known_loader,
                                                                    unknown_loader=train_known_unknown_loader,
                                                                    model=model_wrapper,
                                                                    criterion=criterion,
                                                                    optimizer=optimizer,
                                                                    nb_epoch=epoch,
                                                                    use_msd_net=True,
                                                                    train_phase=True)

            scheduler.step()

            valid_loss, valid_acc_top1, \
            valid_acc_top3, valid_acc_top5 = train_valid_one_epoch(known_loader=valid_known_known_loader,
                                                                  unknown_loader=valid_known_unknown_loader,
                                                                  model=model_wrapper,
                                                                  criterion=criterion,
                                                                  optimizer=optimizer,
                                                                  nb_epoch=epoch,
                                                                  use_msd_net=True,
                                                                  train_phase=False)

        else:
            pass
            # train_loss, train_acc_top1, \
            # train_acc_top3, train_acc_top5 = train_valid_one_epoch(known_loader=train_known_known_loader,
            #                                                        unknown_loader=train_known_unknown_loader,
            #                                                        model=model_wrapper,
            #                                                        criterion=criterion,
            #                                                        optimizer=optimizer,
            #                                                        nb_epoch=epoch,
            #                                                        penalty_factors_known=penalty_factors_for_known,
            #                                                        penalty_factors_unknown=penalty_factors_for_novel,
            #                                                        use_msd_net=False,
            #                                                        train_phase=True)
            #
            # scheduler.step()
            #
            # valid_loss, valid_acc_top1, \
            # valid_acc_top3, valid_acc_top5 = train_valid_one_epoch(known_loader=valid_known_known_loader,
            #                                                        unknown_loader=valid_known_unknown_loader,
            #                                                        model=model_wrapper,
            #                                                        criterion=criterion,
            #                                                        optimizer=optimizer,
            #                                                        nb_epoch=epoch,
            #                                                        penalty_factors_known=penalty_factors_for_known,
            #                                                        penalty_factors_unknown=penalty_factors_for_novel,
            #                                                        use_msd_net=True,
            #                                                        train_phase=False)


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





# TODO: This whole thing needs to be fixed
def test_and_save_probs(test_loader,
                        model,
                        test_type,
                        use_msd_net,
                        epoch_index):
    """

    :param test_loader:
    :param model:
    :param test_unknown:
    :param use_msd_net:
    :return:
    """
    # Setup the paths
    save_known_known_probs_path = save_path_base + "/" + save_path_sub + "/test/known_known/probs_epoch_" + str(epoch_index) + ".npy"
    save_known_known_targets_path = save_path_base + "/" + save_path_sub + "/test/known_known/targets_epoch_" + str(epoch_index) + ".npy"
    save_known_known_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_known/labels_epoch_" + str(epoch_index) + ".npy"
    save_known_known_rt_path = save_path_base + "/" + save_path_sub + "/test/known_known/rts_epoch_" + str(epoch_index) + ".npy"

    save_known_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/probs_epoch_" + str(epoch_index) + ".npy"
    save_known_unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/targets_epoch_" + str(epoch_index) + ".npy"
    save_known_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/labels_epoch_" + str(epoch_index) + ".npy"
    save_known_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/known_unknown/rts_epoch_" + str(epoch_index) + ".npy"

    save_unknown_unknown_probs_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/probs_epoch_" + str(epoch_index) + ".npy"
    save_unknown_unknown_targets_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/targets_epoch_" + str(epoch_index) + ".npy"
    save_unknown_unknown_original_label_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/labels_epoch_" + str(epoch_index) + ".npy"
    save_unknown_unknown_rt_path = save_path_base + "/" + save_path_sub + "/test/unknown_unknown/rts_epoch_" + str(epoch_index) + ".npy"


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
        full_target_list = []
        full_rt_list = []
        full_flops_list = []

        for i in range(len(test_loader)):
            print("*" * 50)

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

            # Check the target labels: keep or change
            # TODO: nb_training_classes or nb_classes?
            for k in range(len(target)):
                if target[k] >= nb_training_classes:
                    target[k] = -1

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # Get the model outputs and RTs
            start =timer()
            output, end_time = model(input_var)

            # TODO: Get the flops for a test image
            flops, _ = measure_model(model, input_var)
            full_flops_list.append(flops)
            # print("HERE")
            # print(flops)
            # sys.exit()

            # Save the RTs
            for end in end_time:
                rts.append(end-start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            # if args.test_with_novel or args.save_probs:
            prob = sm(torch.stack(output).to()) # Shape is [block, batch, class]
            prob_list = np.array(prob.cpu().tolist())
            max_prob = np.max(prob_list)

            # decide whether to do classification or reject
            # When the probability is larger than our threshold
            if max_prob >= thresh_top_1:
                print("Max top-1 probability is %f, larger than threshold %f" % (max_prob, thresh_top_1))

                # Get top-5 predictions from 5 classifiers
                pred_label_list = []
                for j in range(len(output)):
                    _, pred = output[j].data.topk(5, 1, True, True) # pred is a tensor
                    pred_label_list.append(pred.tolist())

                # Update the evaluation metrics for one sample
                # Top-1 and top-5: if any of the 5 classifiers makes a right prediction, consider correct
                # top_5_list = pred_label_list
                top_1_list = []

                for l in pred_label_list:
                    top_1_list.append(l[0][0])

                if target.tolist()[0] in top_1_list:
                    pred_label = target.tolist()[0]
                else:
                    pred_label = top_1_list[-1]

            # When the probability is smaller than our threshold
            else:
                print("Max probability smaller than threshold")
                pred_label = -1

            # Reshape it into [batch, block, class]
            prob_list = np.reshape(prob_list,
                                    (prob_list.shape[1],
                                     prob_list.shape[0],
                                     prob_list.shape[2]))
            target_list = np.array(target.cpu().tolist())

            for one_prob in prob_list.tolist():
                full_prob_list.append(one_prob)
            for one_target in target_list.tolist():
                full_target_list.append(one_target)

        # Save all results to npy file
        full_prob_list_np = np.array(full_prob_list)
        full_target_list_np = np.array(full_target_list)
        full_rt_list_np = np.array(full_rt_list)
        full_original_label_list_np = np.array(full_original_label_list)

        if test_type == "known_known":
            print("Saving probabilities to %s" % save_known_known_probs_path)
            np.save(save_known_known_probs_path, full_prob_list_np)
            print("Saving target labels to %s" % save_known_known_targets_path)
            np.save(save_known_known_targets_path, full_target_list_np)
            print("Saving original labels to %s" % save_known_known_original_label_path)
            np.save(save_known_known_original_label_path, full_original_label_list_np)
            print("Saving RTs to %s" % save_known_known_rt_path)
            np.save(save_known_known_rt_path, full_rt_list_np)

        elif test_type == "known_unknown":
            print("Saving probabilities to %s" % save_known_unknown_probs_path)
            np.save(save_known_unknown_probs_path, full_prob_list_np)
            print("Saving target labels to %s" % save_known_unknown_targets_path)
            np.save(save_known_unknown_targets_path, full_target_list_np)
            print("Saving original labels to %s" % save_known_unknown_original_label_path)
            np.save(save_known_unknown_original_label_path, full_original_label_list_np)
            print("Saving RTs to %s" % save_known_unknown_rt_path)
            np.save(save_known_unknown_rt_path, full_rt_list_np)

        else:
            print("Saving probabilities to %s" % save_unknown_unknown_probs_path)
            np.save(save_unknown_unknown_probs_path, full_prob_list_np)
            print("Saving target labels to %s" % save_unknown_unknown_targets_path)
            np.save(save_unknown_unknown_targets_path, full_target_list_np)
            print("Saving original labels to %s" % save_unknown_unknown_original_label_path)
            np.save(save_unknown_unknown_original_label_path, full_original_label_list_np)
            print("Saving RTs to %s" % save_unknown_unknown_rt_path)
            np.save(save_unknown_unknown_rt_path, full_rt_list_np)


    # TODO: Fix test process for other networks
    else:
        pass




def demo(depth=100,
         growth_rate=12,
         efficient=True):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.
    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)
        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)
        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

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

    valid_transform = train_transform

    test_transform = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize])


    #######################################################################
    # Create dataset and data loader
    #######################################################################
    if use_json_data:
        # Training loaders
        train_known_known_dataset = msd_net_dataset(json_path=train_known_known_path,
                                                    transform=train_transform)
        train_known_known_index = torch.randperm(len(train_known_known_dataset))

        train_known_unknown_dataset = msd_net_dataset(json_path=train_known_unknown_path,
                                                      transform=train_transform)
        train_known_unknown_index = torch.randperm(len(train_known_unknown_dataset))

        train_known_known_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               sampler=torch.utils.data.RandomSampler(train_known_known_index),
                                                               collate_fn=customized_dataloader.collate,
                                                               drop_last=True)
        train_known_unknown_loader = torch.utils.data.DataLoader(train_known_unknown_dataset,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 sampler=torch.utils.data.RandomSampler(train_known_unknown_index),
                                                                 collate_fn=customized_dataloader.collate,
                                                                 drop_last=True)

        # Validation loaders
        valid_known_known_dataset = msd_net_dataset(json_path=valid_known_known_path,
                                                    transform=valid_transform)
        valid_known_known_index = torch.randperm(len(valid_known_known_dataset))

        valid_known_unknown_dataset = msd_net_dataset(json_path=valid_known_unknown_path,
                                                      transform=valid_transform)
        valid_known_unknown_index = torch.randperm(len(valid_known_unknown_dataset))

        valid_known_known_loader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               sampler=torch.utils.data.RandomSampler(valid_known_known_index),
                                                               collate_fn=customized_dataloader.collate,
                                                               drop_last=True)

        valid_known_unknown_loader = torch.utils.data.DataLoader(valid_known_unknown_dataset,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 sampler=torch.utils.data.RandomSampler(valid_known_unknown_index),
                                                                 collate_fn=customized_dataloader.collate,
                                                                 drop_last=True)

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

    else:
        return


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

    # TODO: Maybe adding other networks in the future
    else:
        pass

    # n_flops, n_params = measure_model(model, img_size, img_size)
    # print(n_flops)



    ########################################################################
    # Test-only or Training + validation
    ########################################################################
    # TODO: Fix this testing process + add op count
    if run_test:
        if model_name == "msd_net":
            pass
            # if (use_5_weights==False) and (use_pp_loss==False):
            #     print("Using MSD-Net Base")
            #     index_list = test_msd_base_epoch
            #
            # if (use_5_weights==True) and (use_pp_loss==False):
            #     print("Using 5 weights")
            #     index_list = test_msd_5_weights_epoch
            #
            # if (use_5_weights==True) and (use_pp_loss==True):
            #     print("Using psyphy loss")
            #     index_list = test_msd_pp_epoch
            #
            # for index in index_list:
            #     model_path = save_path_base + "/" + save_path_sub + "/model_epoch_" + str(index) + ".dat"
            #     model.load_state_dict(torch.load(model_path))
            #
            #     print("Loading MSD-Net model:")
            #     print(model_path)

                # print("Testing the known_known samples...")
                # test_and_save_probs(test_loader=test_known_known_loader,
                #                       model=model,
                #                       test_type="known_known",
                #                       use_msd_net=True,
                #                       epoch_index=index)

                # print("Testing the known_unknown samples...")
                # test_and_save_probs(test_loader=test_known_unknown_loader,
                #                     model=model,
                #                     test_type="known_unknown",
                #                     use_msd_net=True,
                #                     epoch_index=index)

                # print("testing the unknown samples...")
                # test_and_save_probs(test_loader=test_unknown_unknown_loader,
                #                       model=model,
                #                       test_type="unknown_unknown",
                #                       use_msd_net=True,
                #                       epoch_index=index)


        else:
            """
            For other networks: there is only one exit,
            so we only need classification accuracy and exit time
            """
            # print("*" * 50)
            # print("Testing the known samples...")
            # test_with_novelty(test_loader=test_known_known_loader,
            #                   model=model,
            #                   test_unknown=False,
            #                   use_msd_net=False)
            #
            # print("*" * 50)
            # print("testing the unknown samples...")
            # test_with_novelty(test_loader=test_unknown_unknown_loader,
            #                   model=model,
            #                   test_unknown=True,
            #                   use_msd_net=False)
            # print("*" * 50)
            pass

        return


    else:

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



def get_pp_factor(rt,
                  scale=1,
                  rt_max=20):
    """
    scalar * (RTmax - RTi) / RTmax + 1

    :param rt:
    :param scale:
    :param rt_max:
    :return:
    """
    if rt > rt_max:
        return 1
    else:
        return (scale*(rt_max-rt)/rt_max +1)



if __name__ == '__main__':
    demo()