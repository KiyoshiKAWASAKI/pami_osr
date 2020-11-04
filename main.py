#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil
import numpy as np
import logging
import csv

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
from itertools import islice
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from timeit import default_timer as timer
import datetime
from utils.customized_dataloader import msd_net_dataset
from utils import customized_dataloader
from utils.psyphy_loss import pp_loss
import torchvision.transforms as transforms
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable


args = arg_parser.parse_args()
torch.manual_seed(args.seed)

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

log_file_path = args.log_file_path

# logging.basicConfig(filename=log_file_path,
#                     level=logging.INFO,
#                     format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
#                     datefmt='%H:%M:%S')

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')


use_5_weights = True
use_pp_loss = False


class train_msdnet():
    def __init__(self,
                 nb_training_classes,

                 grFactor="1-2-4-4", bnFactor='1-2-4-4', nBlocks=5,
                 nChannels=32, base=4, stepmode="even", step=4,
                 growthRate=16, prune="max", bottleneck=True,
                 momentum=0.90, learning_rate=0.1, weight_decay=1e-4):

        self.grFactor = list(map(int, grFactor.split('-')))
        self.bnFactor = list(map(int, bnFactor.split('-')))
        self.nScales = len(grFactor)
        self.nb_training_classes = nb_training_classes
        self.nBlocks = nBlocks
        self.nChannels = nChannels
        self.base = base
        self.stepmode = stepmode
        self.step = step
        self.growthRate = growthRate
        self.prune = prune
        self.bottleneck = bottleneck

        # TODO: This line may be wrong and need to be changed
        self.get_model = getattr(models, 'resnet')(args, nb_blocks=5)
        self.model = torch.nn.DataParallel(self.get_model).cuda()
        self.criterion = nn.CrossEntropyLoss().cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         learning_rate,
                                         momentum=momentum,
                                         weight_decay=weight_decay)





def main():
    global args
    ####################################################################
    # Initialize the model and args
    ####################################################################
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = getattr(models, args.arch)(args, nb_blocks=5)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()


    ####################################################################
    # Define the loss and optimizer
    ####################################################################
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    ####################################################################
    # Define data transformation
    ####################################################################
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

    #####################################################################
    # Create dataset and data loader
    #####################################################################
    # Training loaders
    train_known_known_dataset = msd_net_dataset(json_path=args.train_known_known_path,
                                                transform=train_transform)
    train_known_known_index = torch.randperm(len(train_known_known_dataset))

    train_known_unknown_dataset = msd_net_dataset(json_path=args.train_known_unknown_path,
                                                  transform=train_transform)
    train_known_unknown_index = torch.randperm(len(train_known_unknown_dataset))

    train_known_known_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           sampler=torch.utils.data.RandomSampler(train_known_known_index),
                                                           collate_fn=customized_dataloader.collate,
                                                           drop_last=True)
    train_known_unknown_loader = torch.utils.data.DataLoader(train_known_unknown_dataset,
                                                           batch_size=args.batch_size,
                                                           shuffle=False,
                                                           sampler=torch.utils.data.RandomSampler(train_known_unknown_index),
                                                           collate_fn=customized_dataloader.collate,
                                                           drop_last=True)

    # Validation loaders
    valid_known_known_dataset = msd_net_dataset(json_path=args.valid_known_known_path,
                                                transform=valid_transform)
    valid_known_known_index = torch.randperm(len(valid_known_known_dataset))

    valid_known_unknown_dataset = msd_net_dataset(json_path=args.valid_known_unknown_path,
                                                  transform=valid_transform)
    valid_known_unknown_index = torch.randperm(len(valid_known_unknown_dataset))

    valid_known_known_loader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             sampler=torch.utils.data.RandomSampler(valid_known_known_index),
                                             collate_fn=customized_dataloader.collate,
                                             drop_last=True)

    valid_known_unknown_loader = torch.utils.data.DataLoader(valid_known_unknown_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             sampler=torch.utils.data.RandomSampler(valid_known_unknown_index),
                                             collate_fn=customized_dataloader.collate,
                                             drop_last=True)

    # Test loaders
    test_known_known_dataset = msd_net_dataset(json_path=args.test_known_known_path,
                                               transform=test_transform)
    test_known_known_index = torch.randperm(len(test_known_known_dataset))

    test_known_unknown_dataset = msd_net_dataset(json_path=args.test_known_unknown_path,
                                                 transform=test_transform)
    test_known_unknown_index = torch.randperm(len(test_known_unknown_dataset))

    test_unknown_unknown_dataset = msd_net_dataset(json_path=args.test_unknown_unknown_path,
                                                   transform=test_transform)
    test_unknown_unknown_index = torch.randperm(len(test_unknown_unknown_dataset))

    # When doing test, set the batch size to 1 to test the time one by one accurately
    test_known_known_loader = torch.utils.data.DataLoader(test_known_known_dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             sampler=torch.utils.data.RandomSampler(test_known_known_index),
                                             collate_fn=customized_dataloader.collate,
                                             drop_last=True)

    test_known_unknown_loader = torch.utils.data.DataLoader(test_known_unknown_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              sampler=torch.utils.data.RandomSampler(test_known_unknown_index),
                                              collate_fn=customized_dataloader.collate,
                                              drop_last=True)

    test_unknown_unknown_loader = torch.utils.data.DataLoader(test_unknown_unknown_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            sampler=torch.utils.data.RandomSampler(test_unknown_unknown_index),
                                            collate_fn=customized_dataloader.collate,
                                            drop_last=True)


    ####################################################################
    # Check whether we only do testing
    ####################################################################
    if args.evalmode is not None:
        # print("Doing testing only.")
        logging.info("Doing testing only.")
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            # TODO: add test process
            pass

        else:
            logging.info("Only supporting anytime prediction!")

        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\t'
              'train_prec1\tval_prec1\t'
              'train_prec3\tval_prec3\t'
              'train_prec5\tval_prec5\t']

    ####################################################################
    # Do training and validation: known_known and known_unknown
    ####################################################################
    for epoch in range(args.start_epoch, args.epochs):
        # Adding the option for training early exits using diff penalties.
        if args.train_early_exit:
            logging.info("Training with weighted loss for different classes.")
            """
            Define the penalty factors here:
                For known samples:
                For unknown samples: came from the data distribution for RT
                
            Training strategy:
                Train the model on know_known first, no psyphy-loss applied, only use the simple factors
                Then train the model on known_unknown, use the factor from the distribution and psyphy-loss
            """

            penalty_factors_for_known = [0.2, 0.4, 0.6, 0.8, 1.0]
            penalty_factors_for_novel = [3.897, 5.390, 7.420, 11.491, 22.423]


            if args.switch_batch:
                print("Using the latest training strategy.")

                train_loss, train_prec1, \
                train_prec3, train_prec5, \
                lr = train_known_unknown_switch(train_loader_known=train_known_known_loader,
                                               train_loader_unknown=train_known_unknown_loader,
                                                model=model,
                                                criterion=criterion,
                                                optimizer=optimizer,
                                                epoch=epoch,
                                                penalty_factors_known=penalty_factors_for_known,
                                               penalty_factors_unknown=penalty_factors_for_novel)

                val_loss, val_prec1, \
                val_prec3, val_prec5 = validate_known_unknown_switch(valid_loader_known=valid_known_known_loader,
                                                                        valid_loader_unknown=valid_known_unknown_loader,
                                                                        model=model,
                                                                        criterion=criterion,
                                                                        epoch=epoch,
                                                                        penalty_factors_known=penalty_factors_for_known,
                                                                        penalty_factors_unknown=penalty_factors_for_novel)



        else:
            pass



        ###################################################################
        # Update and save the result
        ###################################################################
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 8).format(epoch, lr, train_loss, val_loss,
                                                              train_prec1, val_prec1,
                                                              train_prec3, val_prec3,
                                                              train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            logging.info('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),}, args, is_best, model_filename, scores)

    logging.info('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    return



############################################################
# Use these 2 for training and validation from 11/01
############################################################
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





def train_known_unknown_switch(train_loader_known,
                                train_loader_unknown,
                                model,
                                criterion,
                                optimizer,
                                epoch,
                                penalty_factors_known,
                                penalty_factors_unknown):

    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param penalty_factors_known:
    :param penalty_factors_unknown:
    :return:
    """

    ##########################################
    # Set up evaluation metrics
    ##########################################
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    model.train()
    end = time.time()

    running_lr = None

    ###################################################
    # training process setup...
    ###################################################
    save_txt_path = os.path.join(args.save, "train_stats_epoch_" + str(epoch) + ".txt")

    # Count number of batches for known and unknown respectively
    nb_known_batches = len(train_loader_known)
    nb_unknown_batches = len(train_loader_unknown)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # Generate index for known and unknown and shuffle
    all_indices = random.sample(list(range(nb_total_batches)), len(list(range(nb_total_batches))))
    print(all_indices)

    known_indices = all_indices[:nb_known_batches]
    print(known_indices)

    unknown_indices = all_indices[nb_known_batches:]
    print(unknown_indices)

    # Create iterator
    known_iter = iter(train_loader_known)
    unknown_iter = iter(train_loader_unknown)

    # Only train one batch for each step
    with open(save_txt_path, 'w') as train_f:
        for i in range(nb_total_batches):
            ##########################################
            # Basic setups
            ##########################################
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=nb_total_batches, method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            loss = 0.0

            ##########################################
            # Get a batch
            ##########################################
            if i in known_indices:
                print("This is a known batch")
                batch = next(known_iter)
                batch_type = "known"

            elif i in unknown_indices:
                print("This is an unknown batch.")
                batch = next(unknown_iter)
                batch_type = "unknown"

            input = batch["imgs"]
            target = batch["labels"] - 1
            rts = batch["rts"]

            input_var = torch.autograd.Variable(input)
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            # Case 1: known batch + 5 weights
            if (batch_type == "known") and (use_5_weights == True):
                print("Case 1")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 2: known batch + no 5 weights
            if (batch_type == "known") and (use_5_weights == False):
                print("Case 2")
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 3: unknown batch + no 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == False) and (use_pp_loss == False):
                print("Case 3")
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 4: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == False):
                print("Case 4")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 5: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == True):
                print("Case 5")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output[j] * penalty_factor
                    scale_factor = get_pp_factor(rts[j])
                    loss += scale_factor * criterion(output_weighted, target_var)


            ##########################################
            # Calculate loss and BP
            ##########################################
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec3, prec5 = accuracy(output[j].data, target_var, topk=(1, 3, 5))
                top1[j].update(prec1.item(), input.size(0))
                top3[j].update(prec3.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@3 {top3.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, nb_total_batches,
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, nb_total_batches,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg, running_lr





def validate_known_unknown_switch(valid_loader_known,
                                  valid_loader_unknown,
                                  model,
                                  criterion,
                                  epoch,
                                  penalty_factors_known,
                                  penalty_factors_unknown):

    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :param penalty_factors_known:
    :param penalty_factors_unknown:
    :return:
    """

    ##########################################
    # Set up evaluation metrics
    ##########################################
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()
    end = time.time()

    ###################################################
    # training process setup...
    ###################################################
    save_txt_path = os.path.join(args.save, "valid_stats_epoch_" + str(epoch) + ".txt")

    # Count number of batches for known and unknown respectively
    nb_known_batches = len(valid_loader_known)
    nb_unknown_batches = len(valid_loader_unknown)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # Generate index for known and unknown and shuffle
    all_indices = random.sample(list(range(nb_total_batches)), len(list(range(nb_total_batches))))
    print(all_indices)

    known_indices = all_indices[:nb_known_batches]
    print(known_indices)

    unknown_indices = all_indices[nb_known_batches:]
    print(unknown_indices)

    # Create iterator
    known_iter = iter(valid_loader_known)
    unknown_iter = iter(valid_loader_unknown)

    # Only train one batch for each step
    with open(save_txt_path, 'w') as f:
        for i in range(nb_total_batches):
            ##########################################
            # Basic setups
            ##########################################
            data_time.update(time.time() - end)
            loss = 0.0

            ##########################################
            # Get a batch
            ##########################################
            if i in known_indices:
                print("This is a known batch")
                batch = next(known_iter)
                batch_type = "known"

            elif i in unknown_indices:
                print("This is an unknown batch.")
                batch = next(unknown_iter)
                batch_type = "unknown"

            input = batch["imgs"]
            target = batch["labels"] - 1
            rts = batch["rts"]

            input_var = torch.autograd.Variable(input)
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            # Case 1: known batch + 5 weights
            if (batch_type == "known") and (use_5_weights == True):
                print("Case 1")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 2: known batch + no 5 weights
            if (batch_type == "known") and (use_5_weights == False):
                print("Case 2")
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 3: unknown batch + no 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == False) and (use_pp_loss == False):
                print("Case 3")
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 4: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == False):
                print("Case 4")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 5: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == True):
                print("Case 5")
                for j in range(len(output)):
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output[j] * penalty_factor
                    scale_factor = get_pp_factor(rts[j])
                    loss += scale_factor * criterion(output_weighted, target_var)


            ##########################################
            # Calculate loss
            ##########################################
            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec3, prec5 = accuracy(output[j].data, target_var, topk=(1, 3, 5))
                top1[j].update(prec1.item(), input.size(0))
                top3[j].update(prec3.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, nb_total_batches,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                f.write('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Acc@1 {top1.val:.4f}\t'
                          'Acc@3 {top3.val:.4f}\t'
                          'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, nb_total_batches,
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg




############################################################
# END
############################################################
def test_with_novelty(val_loader,
                      model,
                      criterion):
    """
    # TODO: Note on 0809 - currently saving everything and do post processing. Need to change in the future.

    1. Using threshold for novelty rejection.
    2. Implementing the early exits.

    :param val_loader:
    :param model:
    :param criterion:
    :return:
    """
    losses = AverageMeter()

    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    # Set the model to evaluation mode
    model.eval()

    # Define the softmax - do softmax to each block.
    sm = torch.nn.Softmax(dim=2)

    full_original_label_list = []
    full_prob_list = []
    full_target_list = []
    full_rt_list = []

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            print("*" * 50)

            rts = []
            input = input.cuda()
            target = target.cuda(async=True)

            # Save original labels to the list
            original_label_list = np.array(target.cpu().tolist())
            for label in original_label_list:
                full_original_label_list.append(label)

            # Check the target labels: keep or change
            if args.test_with_novel:
                for k in range(len(target)):
                    if target[k] >= args.nb_training_classes:
                        target[k] = -1

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # Get the model outputs and RTs
            print("Timer started.")
            start =timer()
            output, end_time = model(input_var)

            # Save the RTs
            for end in end_time:
                rts.append(end-start)
            full_rt_list.append(rts)

            # extract the probability and apply our threshold
            if args.test_with_novel or args.save_probs:
                prob = sm(torch.stack(output).to()) # Shape is [block, batch, class]
                prob_list = np.array(prob.cpu().tolist())
                max_prob = np.max(prob_list)

                # decide whether to do classification or reject
                # When the probability is larger than our threshold
                if max_prob >= args.thresh_top_1:
                    print("Max top-1 probability is %f, larger than threshold %f" % (max_prob, args.thresh_top_1))

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
                    pred_label = -1


            if args.save_probs:
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

            if not isinstance(output, list):
                output = [output]


            # TODO (novelty-rejection): torch cannot deal with label -1
            # if not args.test_with_novel:
            #     loss = 0.0
            #     for j in range(len(output)):
            #         loss += criterion(output[j], target_var)
            #
            #     losses.update(loss.item(), input.size(0))



            # TODO: getting evaluations??
            if args.test_with_novel:
                pass

            else:
                pass


        if args.save_probs == True:
            full_prob_list_np = np.array(full_prob_list)
            full_target_list_np = np.array(full_target_list)
            full_rt_list_np = np.array(full_rt_list)
            full_original_label_list_np = np.array(full_original_label_list)

            print("Saving probabilities to %s" % args.save_probs_path)
            np.save(args.save_probs_path, full_prob_list_np)
            print("Saving target labels to %s" % args.save_targets_path)
            np.save(args.save_targets_path, full_target_list_np)
            print("Saving original labels to %s" % args.save_original_label_path)
            np.save(args.save_original_label_path, full_original_label_list_np)
            print("Saving RTs to %s" % args.save_rt_path)
            np.save(args.save_rt_path, full_rt_list_np)



    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))

    return losses.avg, top1[-1].avg, top5[-1].avg




def save_checkpoint(state, args, is_best, filename, result):
    # print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    logging.info("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return




def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    logging.info("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    logging.info("=> loaded checkpoint '{}'".format(model_filename))
    return state



class AverageMeter(object):
    """Computes and stores the average and current value"""

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
    Computes the precor@k for the specified values of k
    :param output:
    :param target:
    :param topk:
    :return:
    """

    maxk = max(topk)
    batch_size = target.size(0)

    # print("Here are the ground truth label")
    # print(target)

    _, pred = output.topk(maxk, 1, True, True)
    # print("Here is the pred in accuracy function")
    # # print(pred.shape) # torch.Size ==> [batch_size, nb_clfs]
    # print(pred)

    pred = pred.t()

    try:
        correct = pred.eq(target.view(1, -1).expand_as(pred))


        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res

    except:
        print("Error occured ")
        return [0.0, 0.0, 0.0]




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



def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


if __name__ == '__main__':
    main()
