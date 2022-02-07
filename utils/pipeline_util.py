# Functions for pipeline: training, testing, features

import os
import time
import torch
import numpy as np
from timeit import default_timer as timer
import sys
import warnings
warnings.filterwarnings("ignore")
import random
import math
from tqdm import tqdm



def train_valid_test_one_epoch_for_known(args,
                                       loader_with_rt,
                                       loader_without_rt,
                                       model,
                                       criterion,
                                       optimizer,
                                       nb_epoch,
                                       use_msd_net,
                                       train_phase,
                                       save_path,
                                       use_performance_loss,
                                       use_exit_loss,
                                       cross_entropy_weight,
                                       perform_loss_weight,
                                       exit_loss_weight,
                                       known_exit_rt=None,
                                       known_thresholds=None,
                                       debug=False,
                                       model_name="msd_net",
                                       nb_clfs=5,
                                       nBlocks=5,
                                       nb_rt_classes=40,
                                       nb_no_rt_classes=253,
                                       nb_classes=294,
                                       rt_max=28,
                                       nb_sample_per_bacth=16,
                                       human_known_rt_max=28,
                                       human_unknown_rt_max=28,
                                       machine_known_rt_max=0.057930,
                                       machine_unknown_rt_max=0.071147):

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
    else:
        save_txt_path = os.path.join(save_path, "valid_stats_epoch_" + str(nb_epoch) + ".txt")

    # Count number of batches for known and unknown respectively
    nb_rt_batches = len(loader_with_rt)
    nb_no_rt_batches = len(loader_without_rt)
    nb_total_batches = nb_rt_batches + nb_no_rt_batches

    print("There are %d batches in RT loader" % nb_rt_batches)
    print("There are %d batches in no RT loader" % nb_no_rt_batches)

    # Generate index for known and unknown and shuffle
    all_indices = random.sample(list(range(nb_total_batches)), len(list(range(nb_total_batches))))
    rt_indices = all_indices[:nb_rt_batches]
    no_rt_indices = all_indices[nb_rt_batches:]

    # Create iterator
    rt_iter = iter(loader_with_rt)
    no_rt_iter = iter(loader_without_rt)

    # Only train one batch for each step
    with open(save_txt_path, 'w') as f:
        for i in tqdm(range(nb_total_batches)):
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
            if i in rt_indices:
                batch = next(rt_iter)

            elif i in no_rt_indices:
                try:
                    batch = next(no_rt_iter)
                except:
                    continue

            input = batch["imgs"]
            rts = batch["rts"]
            target = batch["labels"]

            # Convert into PyTorch tensor
            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            start = timer()
            output, feature, end_time = model(input_var)

            full_rt_list = []
            for end in end_time[0]:
                full_rt_list.append(end - start)

            # len(output) = 5
            if not isinstance(output, list):
                output = [output]

            # TODO: need to double check this section when using psyphy
            if use_exit_loss == True:
                ##########################################
                # Get exits for each sample
                ##########################################
                # Define the RT cuts and the thresholds
                exit_rt_cut = known_exit_rt
                top_1_threshold = known_thresholds

                """
                Find the target exit RT for each sample according to its RT:
                    If a batch has human RT: check the 5 intervals from human RT distribution
                    If a batch doesn't have human RT: assign zeroes
                """
                target_exit_rt = []

                if rts[0] != 0:
                    for one_rt in rts:
                        if (one_rt < exit_rt_cut[0]):
                            target_exit_rt.append(exit_rt_cut[0])
                        if (one_rt >= exit_rt_cut[0]) and (one_rt < exit_rt_cut[1]):
                            target_exit_rt.append(exit_rt_cut[1])
                        if (one_rt >= exit_rt_cut[1]) and (one_rt < exit_rt_cut[2]):
                            target_exit_rt.append(exit_rt_cut[2])
                        if (one_rt >= exit_rt_cut[2]) and (one_rt < exit_rt_cut[3]):
                            target_exit_rt.append(exit_rt_cut[3])
                        if (one_rt >= exit_rt_cut[3]) and (one_rt < exit_rt_cut[4]):
                            target_exit_rt.append(exit_rt_cut[4])
                else:
                    target_exit_rt = [0.0] * nb_sample_per_bacth

                if debug:
                    print("Human RTs from batch:")
                    print(rts)
                    print("Exit RT cut:")
                    print(exit_rt_cut)
                    print("Obtained target exit RT")
                    print(target_exit_rt)

                """
                Find the actual/predicted RT for each sample

                Case 1:
                    prob > threshold && prediction is correct - exit right away
                Case 2:
                    prob < threshold && not at the last exit 
                    or
                    prob > threshold but predicition is wrong - check next exit
                Case 3:
                    prob < threshold && at the last exit - exit no matter what
                """
                full_prob_list = []

                # Logits to probs: Extract the probability and apply our threshold
                sm = torch.nn.Softmax(dim=2)

                prob = sm(torch.stack(output).to())  # Shape is [block, batch, class]
                prob_list = np.array(prob.cpu().tolist())

                # Reshape it into [batch, block, class]
                prob_list = np.reshape(prob_list,
                                       (prob_list.shape[1],
                                        prob_list.shape[0],
                                        prob_list.shape[2]))

                for one_prob in prob_list.tolist():
                    full_prob_list.append(one_prob)

                # Thresholding - check for each exit
                pred_exit_rt = []

                for k in range(len(full_prob_list)):
                    # Get probs and GT labels
                    prob = full_prob_list[k]
                    gt_label = target[k]

                    # check each classifier in order and decide when to exit
                    for j in range(nb_clfs):
                        one_prob = prob[j]
                        max_prob = np.sort(one_prob)[-1]
                        pred = np.argmax(one_prob)

                        # If this is not the last classifier
                        if j != nb_clfs - 1:
                            # Updated - use different threshold for each exit
                            if (max_prob > top_1_threshold[j]) and (pred == gt_label):
                                # Case 1
                                pred_rt = full_rt_list[j]
                                pred_exit_rt.append(pred_rt)
                                break
                            else:
                                # Case 2
                                continue
                        # Case 3
                        else:
                            pred_rt = full_rt_list[-1]
                            pred_exit_rt.append(pred_rt)

                # Check the human RTs and machine RTs
                if debug:
                    print("Machine RT:")
                    print(pred_exit_rt)

            ##########################################
            # Only MSD-Net
            ##########################################
            if model_name == "msd_net":
                for j in range(len(output)):
                    # Part 1: Cross-entropy loss
                    ce_loss = criterion(output[j], target_var)

                    # Part 2: Performance psyphy loss
                    try:
                        perform_loss = get_perform_loss(rt=rts[j], rt_max=rt_max)
                    except:
                        perform_loss = 0.0

                    # Part 3: Exit psyphy loss
                    if use_exit_loss:
                        try:
                            exit_loss = get_exit_loss(pred_exit_rt=pred_exit_rt[j],
                                                      target_exit_rt=target_exit_rt[j],
                                                      human_known_rt_max=human_known_rt_max,
                                                      human_unknown_rt_max=human_unknown_rt_max,
                                                      machine_known_rt_max=machine_known_rt_max,
                                                      machine_unknown_rt_max=machine_unknown_rt_max,
                                                      batch_type=None)
                        except:
                            exit_loss = 0.0

                    # 3 Cases
                    if (use_performance_loss == True) and (use_exit_loss == False):
                        loss += cross_entropy_weight * ce_loss + \
                                perform_loss_weight * perform_loss

                    if (use_exit_loss == True) and (use_exit_loss == True):
                        loss += cross_entropy_weight * ce_loss + \
                                perform_loss_weight * perform_loss + \
                                exit_loss_weight * exit_loss

                    if (use_performance_loss == False) and (use_exit_loss == False):
                        loss += ce_loss


            else:
                # TODO(low priority): other networks -
                #  may be the same with MSD Net cause 5 weights are gone?
                pass

            ##########################################
            # Calculate loss and BP
            ##########################################
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)): # shape of output: [nb_exit, batch, nb_classes]
                # Calculate acc for all classes
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


            # TODO: Update resultes with RT and without RT separately
            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f}\t'
                    'Acc@1 {top1.val:.4f}\t'
                    'Acc@3 {top3.val:.4f}\t'
                    'Acc@5 {top5.val:.4f}\n'.format(
                nb_epoch, i + 1, nb_total_batches,
                loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg





def train_valid_test_one_epoch(args,
                               known_loader,
                               unknown_loader,
                               model,
                               criterion,
                               optimizer,
                               nb_epoch,
                               use_msd_net,
                               train_phase,
                               save_path,
                               use_performance_loss,
                               use_exit_loss,
                               cross_entropy_weight,
                               perform_loss_weight,
                               exit_loss_weight,
                               known_exit_rt=None,
                               unknown_exit_rt=None,
                               known_thresholds=None,
                               unknown_thresholds=None,
                               debug=False,
                               train_binary=False,
                               model_name="msd_net",
                               nb_clfs=5,
                               nBlocks=5,
                               nb_classes=294,
                               nb_training_classes=294,
                               rt_max=28,
                               nb_sample_per_batch=16,
                               human_known_rt_max=28,
                               human_unknown_rt_max=28,
                               machine_known_rt_max=0.057930,
                               machine_unknown_rt_max=0.071147):

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
        for i in tqdm(range(nb_total_batches)):
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
                batch_type = "known"

            elif i in unknown_indices:
                try:
                    batch = next(unknown_iter)
                    batch_type = "unknown"
                except:
                    continue

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

            # Adjust the label for unknown
            if batch_type == "unknown":
                for i in range(len(target)):
                    target[i] = nb_training_classes - 1

            # Convert into PyTorch tensor
            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            start = timer()
            output, feature, end_time = model(input_var)

            full_rt_list = []
            for end in end_time[0]:
                full_rt_list.append(end - start)

            # len(output) = 5
            if not isinstance(output, list):
                output = [output]

            if use_exit_loss == True:
                ##########################################
                # TODO: Get exits for each sample
                ##########################################
                # TODO: Define the RT cuts for known and unknown, and the thresholds
                if batch_type == "known":
                    exit_rt_cut = known_exit_rt
                    top_1_threshold = known_thresholds
                elif batch_type == "unknown":
                    exit_rt_cut = unknown_exit_rt
                    top_1_threshold = unknown_thresholds
                else:
                    print("Unknown batch type!!")
                    sys.exit()

                if debug:
                    print("batch_type: %s" % batch_type)

                """
                Find the target exit RT for each sample according to its RT:
                    If a batch has human RT: check the 5 intervals from human RT distribution
                    If a batch doesn't have human RT: assign zeroes
                """
                target_exit_rt = []

                if rts[0] != 0:
                    for one_rt in rts:
                        if (one_rt < exit_rt_cut[0]):
                            target_exit_rt.append(exit_rt_cut[0])
                        if (one_rt >= exit_rt_cut[0]) and (one_rt < exit_rt_cut[1]):
                            target_exit_rt.append(exit_rt_cut[1])
                        if (one_rt >= exit_rt_cut[1]) and (one_rt < exit_rt_cut[2]):
                            target_exit_rt.append(exit_rt_cut[2])
                        if (one_rt >= exit_rt_cut[2]) and (one_rt < exit_rt_cut[3]):
                            target_exit_rt.append(exit_rt_cut[3])
                        if (one_rt >= exit_rt_cut[3]) and (one_rt < exit_rt_cut[4]):
                            target_exit_rt.append(exit_rt_cut[4])
                else:
                    target_exit_rt = [0.0] * nb_sample_per_batch

                if debug:
                    print("Human RTs from batch:")
                    print(rts)
                    print("Exit RT cut:")
                    print(exit_rt_cut)
                    print("Obtained target exit RT")
                    print(target_exit_rt)

                """
                Find the actual/predicted RT for each sample

                Case 1:
                    prob > threshold && prediction is correct - exit right away
                Case 2:
                    prob < threshold && not at the last exit 
                    or
                    prob > threshold but predicition is wrong - check next exit
                Case 3:
                    prob < threshold && at the last exit - exit no matter what
                """
                full_prob_list = []

                # Logits to probs: Extract the probability and apply our threshold
                sm = torch.nn.Softmax(dim=2)

                prob = sm(torch.stack(output).to())  # Shape is [block, batch, class]
                prob_list = np.array(prob.cpu().tolist())

                # Reshape it into [batch, block, class]
                prob_list = np.reshape(prob_list,
                                       (prob_list.shape[1],
                                        prob_list.shape[0],
                                        prob_list.shape[2]))

                for one_prob in prob_list.tolist():
                    full_prob_list.append(one_prob)

                # Thresholding - check for each exit
                pred_exit_rt = []

                for i in range(len(full_prob_list)):
                    # Get probs and GT labels
                    prob = full_prob_list[i]
                    gt_label = target[i]

                    # check each classifier in order and decide when to exit
                    for j in range(nb_clfs):
                        one_prob = prob[j]
                        max_prob = np.sort(one_prob)[-1]
                        pred = np.argmax(one_prob)

                        # If this is not the last classifier
                        if j != nb_clfs - 1:
                            # Updated - use different threshold for each exit
                            if (max_prob > top_1_threshold[j]) and (pred == gt_label):
                                # Case 1
                                pred_rt = full_rt_list[j]
                                pred_exit_rt.append(pred_rt)
                                break
                            else:
                                # Case 2
                                continue
                        # Case 3
                        else:
                            pred_rt = full_rt_list[-1]
                            pred_exit_rt.append(pred_rt)

                # Check the human RTs and machine RTs
                if debug:
                    print("Machine RT:")
                    print(pred_exit_rt)

            ##########################################
            # Only MSD-Net
            ##########################################
            if model_name == "msd_net":
                for j in range(len(output)):
                    # Part 1: Cross-entropy loss
                    ce_loss = criterion(output[j], target_var)

                    # Part 2: Performance psyphy loss
                    try:
                        perform_loss = get_perform_loss(rt=rts[j], rt_max=rt_max)
                    except:
                        perform_loss = 0.0

                    # Part 3: Exit psyphy loss
                    if use_exit_loss:
                        try:
                            exit_loss = get_exit_loss(pred_exit_rt=pred_exit_rt[j],
                                                      target_exit_rt=target_exit_rt[j],
                                                      human_known_rt_max=human_known_rt_max,
                                                      human_unknown_rt_max=human_unknown_rt_max,
                                                      machine_known_rt_max=machine_known_rt_max,
                                                      machine_unknown_rt_max=machine_unknown_rt_max,
                                                      batch_type=batch_type)
                        except:
                            exit_loss = 0.0

                    # 3 Cases
                    if (use_performance_loss == True) and (use_exit_loss == False):
                        # print("Using cross-entropy and performance loss")
                        loss += cross_entropy_weight * ce_loss + \
                                perform_loss_weight * perform_loss

                    if (use_exit_loss == True) and (use_exit_loss == True):
                        # print("Using all 3 losses")
                        loss += cross_entropy_weight * ce_loss + \
                                perform_loss_weight * perform_loss + \
                                exit_loss_weight * exit_loss

                    if (use_performance_loss == False) and (use_exit_loss == False):
                        # print("Using cross-entropy only")
                        loss += ce_loss


            else:
                # TODO(low priority): other networks -
                #  may be the same with MSD Net cause 5 weights are gone?
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

            f.write('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss.val:.4f}\t'
                    'Acc@1 {top1.val:.4f}\t'
                    'Acc@3 {top3.val:.4f}\t'
                    'Acc@5 {top5.val:.4f}\n'.format(
                nb_epoch, i + 1, nb_total_batches,
                loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg




def save_probs_and_features(test_loader,
                            model,
                            test_type,
                            use_msd_net,
                            epoch_index,
                            npy_save_dir,
                            part_index=None):
    """
    batch size is always one for testing.

    :param test_loader:
    :param model:
    :param test_unknown:
    :param use_msd_net:
    :return:
    """

    # Set the model to evaluation mode
    model.cuda()
    model.eval()

    if use_msd_net:
        print("Testing MSD-Net...")
        sm = torch.nn.Softmax(dim=2)

        # For MSD-Net, save everything into npy files
        full_original_label_list = []
        full_prob_list = []
        full_rt_list = []
        full_feature_list = []

        print(len(test_loader))
        # sys.exit()

        for i in tqdm(range(len(test_loader))):
            try:
                batch = next(iter(test_loader))
            except:
                continue

            input = batch["imgs"]
            target = batch["labels"]

            rts = []
            input = input.cuda()
            target = target.cuda(async=True)

            # Save original labels to the list
            original_label_list = np.array(target.cpu().tolist())
            for label in original_label_list:
                full_original_label_list.append(label)

            input_var = torch.autograd.Variable(input)

            # Get the model outputs and RTs
            start =timer()
            output, feature, end_time = model(input_var)

            # Handle the features
            feature = feature[0][0].cpu().detach().numpy()
            feature = np.reshape(feature, (1, feature.shape[0] * feature.shape[1] * feature.shape[2]))

            for one_feature in feature.tolist():
                full_feature_list.append(one_feature)

            # Save the RTs
            for end in end_time[0]:
                # print("Processes one sample in %f sec" % (end - start))
                rts.append(end-start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            prob = sm(torch.stack(output).to()) # Shape is [block, batch, class]
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

        if part_index is not None:
            save_prob_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_probs.npy"
            save_label_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_labels.npy"
            save_rt_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_rts.npy"
            save_feature_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_part_" + str(part_index) + "_features.npy"

        else:
            save_prob_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_probs.npy"
            save_label_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_labels.npy"
            save_rt_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_rts.npy"
            save_feature_path = npy_save_dir + "/" + test_type + "_epoch_" + str(epoch_index) + "_features.npy"

        print("Saving probabilities to %s" % save_prob_path)
        np.save(save_prob_path, full_prob_list_np)
        print("Saving original labels to %s" % save_label_path)
        np.save(save_label_path, full_original_label_list_np)
        print("Saving RTs to %s" % save_rt_path)
        np.save(save_rt_path, full_rt_list_np)
        print("Saving features to %s" % save_feature_path)
        np.save(save_feature_path, full_feature_list)


    # TODO: Test process for other networks - is it different??
    else:
        pass



def find_best_model(dir_to_models,
                    model_format=".dat"):
    """
    Find the best model in a directory

    :param dir_to_models:
    :return:
    """
    # List all the models
    all_models = []

    for file in os.listdir(dir_to_models):
        if file.endswith(model_format):
            all_models.append(os.path.join(dir_to_models, file))

    # Find the index for best model
    all_indices = []

    for one_model in all_models:
        one_index = int(one_model.split("/")[-1].split(".")[0].split("_")[-1])
        all_indices.append(one_index)

    best_epoch = max(all_indices)

    best_model_path = dir_to_models + "/model_epoch_" + str(best_epoch) + model_format

    return best_epoch, best_model_path




def get_perform_loss(rt,
                     rt_max,
                     use_addition=True):
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
        if use_addition:
            return ((rt_max-rt)/rt_max)
        else:
            return ((rt_max-rt)/rt_max + 1)




def get_exit_loss(pred_exit_rt,
                  target_exit_rt,
                  human_known_rt_max,
                  human_unknown_rt_max,
                  machine_known_rt_max,
                  machine_unknown_rt_max,
                  batch_type=None):
    """

    :param pred_exit_rt:
    :param target_exit_rt:
    :param human_known_rt_max:
    :param human_unknown_rt_max:
    :param machine_known_rt_max:
    :param machine_unknown_rt_max:
    :param batch_type:
    :return:
    """
    if batch_type is not None:
        if batch_type == "known":
            exit_loss = abs((target_exit_rt/human_known_rt_max) - (pred_exit_rt/machine_known_rt_max))
        elif batch_type == "unknown":
            exit_loss = abs((target_exit_rt/human_unknown_rt_max) - (pred_exit_rt/machine_unknown_rt_max))
        else:
            sys.exit()
    else:
        exit_loss = abs((target_exit_rt / human_known_rt_max) - (pred_exit_rt / machine_known_rt_max))

    return exit_loss




class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/
    imagenet/main.py
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