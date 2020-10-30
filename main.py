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

# TODO: Use a small json file for debugging first
debugging_json_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/train_known_unknown.json"



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

# Define the TensorBoard
# writer = SummaryWriter(args.tf_board_path)


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

    # print(model.named_parameters())
    # sys.exit()

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
            # Test 3 diff categories separately
            test_with_novelty_pp(data_loader=test_known_known_loader,
                                  model=model,
                                  criterion=criterion)

            test_with_novelty_pp(data_loader=test_known_unknown_loader,
                                  model=model,
                                  criterion=criterion)

            test_with_novelty_pp(data_loader=test_unknown_unknown_loader,
                                  model=model,
                                  criterion=criterion)

        else:
            logging.info("Only supporting anytime prediction!")

        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\t'
              'train_prec1_known\tval_prec1_known\t'
              'train_prec3_known\tval_prec3_known\t'
              'train_prec5_known\tval_prec5_known\t'
              'train_prec1_unknown\tval_prec1_unknown\t'
              'train_prec3_unknown\tval_prec3_unknown\t'
              'train_prec5_unknown\tval_prec5_unknown\t']

    ####################################################################
    # Do training and validation: known_known and known_unknown
    ####################################################################
    # Get the step count: for logging tensor board
    nb_known_batches = len(train_known_known_loader)
    nb_unknown_batches = len(train_known_unknown_loader)

    # Check which category has more samples/batches
    if nb_known_batches >= nb_unknown_batches:
        base_step = len(train_known_known_loader)
    else:
        base_step = len(train_known_unknown_loader)



    for epoch in range(args.start_epoch, args.epochs):
        # Adding the option for training k+1 classes
        if args.train_k_plus_1:
            return
            # logging.info("Training MSD-Net on K+1 classes.")
            # train_loss, train_prec1, train_prec3, train_prec5, lr = train_k_plus_one(train_loader, model, criterion, optimizer, epoch)
            # val_loss, val_prec1, val_prec3, val_prec5 = validate_k_plus_one(val_loader, model, criterion, epoch)

        # Adding the option for training early exits using diff penalties.
        elif args.train_early_exit:
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

            if args.train_in_order:
                print("Training known and unknown batches in order")

                train_loss, train_prec1_known, train_prec3_known, train_prec5_known, \
                train_prec1_unknown, train_prec3_unknown, train_prec5_unknown, lr = train_known_unknown_in_order(train_loader_known=train_known_known_loader,
                                                                                                     train_loader_unknown=train_known_unknown_loader,
                                                                                                    model=model,
                                                                                                    criterion=criterion,
                                                                                                    optimizer=optimizer,
                                                                                                    epoch=epoch,
                                                                                                    penalty_factors_known=penalty_factors_for_known,
                                                                                                    penalty_factors_unknown=penalty_factors_for_novel)

                val_loss, val_prec1_known, val_prec3_known, val_prec5_known, \
                val_prec1_unknown, val_prec3_unknown, val_prec5_unknown = valid_known_unknown_in_order(valid_loader_known=valid_known_known_loader,
                                                                                         valid_loader_unknown=valid_known_unknown_loader,
                                                                                         model=model,
                                                                                         criterion=criterion,
                                                                                         penalty_factors_known=penalty_factors_for_known,
                                                                                         penalty_factors_unknown=penalty_factors_for_novel,
                                                                                         epoch=epoch,
                                                                                         base_step=base_step)

            elif args.use_first_version:
                train_loss, train_prec1, train_prec3, train_prec5, lr = train_early_exit_with_pp_loss(train_loader_known=train_known_known_loader,
                                                                                                      train_loader_unknown=train_known_unknown_loader,
                                                                                                      model=model,
                                                                                                      criterion=criterion,
                                                                                                      optimizer=optimizer,
                                                                                                      epoch=epoch,
                                                                                                      penalty_factors_known=penalty_factors_for_known,
                                                                                                      penalty_factors_unknown=penalty_factors_for_novel)

                val_loss, val_prec1, val_prec3, val_prec5, val_prec1, val_prec3, val_prec5 = validate_early_exit_with_pp_loss(val_known_loader=valid_known_known_loader,
                                                                                             val_unknown_loader=valid_known_unknown_loader,
                                                                                             model=model,
                                                                                             criterion=criterion,
                                                                                             penalty_factors_known=penalty_factors_for_known,
                                                                                             penalty_factors_unknown=penalty_factors_for_novel)

            elif args.train_known_only:
                train_loss, train_prec1_known, train_prec3_known, train_prec5_known, \
                train_prec1_unknown, train_prec3_unknown, train_prec5_unknown, lr = train_known_classes(train_loader_known=train_known_known_loader,
                                                                                                        model=model,
                                                                                                        criterion=criterion,
                                                                                                        optimizer=optimizer,
                                                                                                        epoch=epoch,
                                                                                                        penalty_factors_known=penalty_factors_for_known)

                val_loss, val_prec1_known, val_prec3_known, val_prec5_known, \
                val_prec1_unknown, val_prec3_unknown, val_prec5_unknown = valid_known_classes(valid_loader_known=valid_known_known_loader,
                                                                                            model=model,
                                                                                            criterion=criterion,
                                                                                            epoch=epoch,
                                                                                            base_step=base_step)


            else:
                pass



        ###################################################################
        # Update and save the result
        ###################################################################
        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 14).format(epoch, lr, train_loss, val_loss,
                                                             train_prec1_known, val_prec1_known,
                                                             train_prec3_known, val_prec3_known,
                                                             train_prec5_known, val_prec5_known,
                                                             train_prec1_unknown, val_prec1_unknown,
                                                             train_prec3_unknown, val_prec3_unknown,
                                                             train_prec5_unknown, val_prec5_unknown))

        is_best = val_prec1_known > best_prec1
        if is_best:
            best_prec1 = val_prec1_known
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
# Use these 2 for training and validation from 09/15
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







def train_known_classes(train_loader_known,
                      model,
                      criterion,
                      optimizer,
                      epoch,
                      penalty_factors_known,
                      base_step):
    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion_known:
    :param criterion_unknown:
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

    # Initialize the accuracy for known and unknown respectively
    top1_known, top3_known, top5_known = [], [], []
    for i in range(args.nBlocks):
        top1_known.append(AverageMeter())
        top3_known.append(AverageMeter())
        top5_known.append(AverageMeter())

    model.train()
    end = time.time()

    running_lr = None

    ###################################################
    # training known data: no RT, just normal training
    ###################################################
    save_known_txt_path = os.path.join(args.save, "train_known_stats_epoch_" + str(epoch) + ".txt")

    with open(save_known_txt_path, 'w') as train_f:
        for i, batch in enumerate(train_loader_known):
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=len(train_loader_known), method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            loss = 0.0

            known_input = batch["imgs"]
            known_target = batch["labels"] - 1

            known_input_var = torch.autograd.Variable(known_input)
            known_target = known_target.cuda(async=True)
            known_target_var = torch.autograd.Variable(known_target).long()

            output_1 = model(known_input_var)

            if not isinstance(output_1, list):
                output_1 = [output_1]

            for j in range(len(output_1)):
                if args.use_5_weights:
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output_1[j] * penalty_factor
                else:
                    output_weighted = output_1[j]

                loss += criterion(output_weighted, known_target_var)

            losses.update(loss.item(), known_input.size(0))

            ##########################################
            # Evaluate model on known samples
            ##########################################
            for j in range(len(output_1)):
                prec1_known, prec3_known, prec5_known = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                top1_known[j].update(prec1_known.item(), known_input.size(0))
                top3_known[j].update(prec3_known.item(), known_input.size(0))
                top5_known[j].update(prec5_known.item(), known_input.size(0))

            ###################################################
            # BP
            ###################################################
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
                             'Acc known@1 {top1.val:.4f}\t'
                             'Acc known@3 {top3.val:.4f}\t'
                             'Acc known@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_known),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc known@1 {top1.val:.4f}\t'
                              'Acc known@3 {top3.val:.4f}\t'
                              'Acc known@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_known),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))

    top1_unknown = 0.0
    top3_unknown = 0.0
    top5_unknown = 0.0

    return losses.avg, top1_known[-1].avg, top3_known[-1].avg, top5_known[-1].avg, \
           top1_unknown, top3_unknown, top5_unknown, running_lr





def valid_known_classes(valid_loader_known,
                          model,
                          criterion,
                          epoch,
                          penalty_factors_known,
                          base_step):
    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion_known:
    :param criterion_unknown:
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
    save_txt_path = os.path.join(args.save, "valid_stats_epoch_" + str(epoch) + ".txt")

    with torch.no_grad():
        with open(save_txt_path, 'w') as train_f:
            for i, batch in enumerate(valid_loader_known):
                data_time.update(time.time() - end)

                loss = 0.0

                ###################################################
                # training known data: no RT, just normal training
                ###################################################
                # print("Training the first batch")
                known_input = batch["imgs"]
                known_target = batch["labels"] - 1

                known_input_var = torch.autograd.Variable(known_input)
                known_target = known_target.cuda(async=True)
                known_target_var = torch.autograd.Variable(known_target).long()

                output_1 = model(known_input_var)

                if not isinstance(output_1, list):
                    output_1 = [output_1]

                for j in range(len(output_1)):
                    if args.use_5_weights:
                        penalty_factor = penalty_factors_known[j]
                        output_weighted = output_1[j] * penalty_factor
                    else:
                        output_weighted = output_1[j]

                    loss += criterion(output_weighted, known_target_var)

                losses.update(loss.item(), known_input.size(0))


                ##########################################
                # Evaluate model
                ##########################################
                for j in range(len(output_1)):
                    prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                    top1[j].update(prec1.item(), known_input.size(0))
                    top3[j].update(prec3.item(), known_input.size(0))
                    top5[j].update(prec5.item(), known_input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


                if i % args.print_freq == 0:
                    # Implement TensorBoard
                    # writer.add_scalar('validation loss', losses.val, i*epoch+base_step)
                    # writer.add_scalar('valid acc top-1', top1[-1].val, i*epoch+base_step)
                    # writer.add_scalar('valid acc top-3', top3[-1].val, i*epoch+base_step)
                    # writer.add_scalar('valid acc top-5', top5[-1].val, i*epoch+base_step)

                    logging.info('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Acc@1 {top1.val:.4f}\t'
                          'Acc@3 {top3.val:.4f}\t'
                          'Acc@5 {top5.val:.4f}\n'.format(
                            epoch, i + 1, len(valid_loader_known),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                    train_f.write('Epoch: [{0}][{1}/{2}]\t'
                                  'Time {batch_time.avg:.3f}\t'
                                  'Data {data_time.avg:.3f}\t'
                                  'Loss {loss.val:.4f}\t'
                                  'Acc@1 {top1.val:.4f}\t'
                                  'Acc@3 {top3.val:.4f}\t'
                                  'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(valid_loader_known),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    top1_unknown = 0.0
    top3_unknown = 0.0
    top5_unknown = 0.0

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg, top1_unknown, top3_unknown, top5_unknown




def train_known_unknown_switch(train_loader_known,
                                train_loader_unknown,
                                model,
                                criterion,
                                optimizer,
                                epoch,
                                penalty_factors_known,
                                base_step):

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

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # Generate index for known and unknown and shuffle
    all_indices = random.shuffle(list(range(nb_known_batches+nb_unknown_batches)))

    known_indices = all_indices[:nb_known_batches]
    unknown_indices = all_indices[nb_known_batches:]

    print(known_indices)
    print(unknown_indices)

    #

    with open(save_txt_path, 'w') as train_f:
        for i, batch in enumerate(train_loader_known):
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=len(train_loader_known), method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            loss = 0.0

            ###################################################
            # training known data: no RT, just normal training
            ###################################################
            # print("Training the first batch")
            known_input = batch["imgs"]
            known_target = batch["labels"] - 1

            known_input_var = torch.autograd.Variable(known_input)
            known_target = known_target.cuda(async=True)
            known_target_var = torch.autograd.Variable(known_target).long()

            output_1 = model(known_input_var)

            if not isinstance(output_1, list):
                output_1 = [output_1]

            for j in range(len(output_1)):
                if args.use_5_weights:
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output_1[j] * penalty_factor
                else:
                    output_weighted = output_1[j]

                loss += criterion(output_weighted, known_target_var)

            losses.update(loss.item(), known_input.size(0))


            ##########################################
            # Evaluate model
            ##########################################
            for j in range(len(output_1)):
                prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                top1[j].update(prec1.item(), known_input.size(0))
                top3[j].update(prec3.item(), known_input.size(0))
                top5[j].update(prec5.item(), known_input.size(0))

            ###################################################
            # BP
            ###################################################
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if i % args.print_freq == 0:
                # Implement TensorBoard
                writer.add_scalar('training loss', losses.val, i*epoch+base_step)
                writer.add_scalar('train acc top-1', top1[-1].val, i*epoch+base_step)
                writer.add_scalar('train acc top-3', top3[-1].val, i*epoch+base_step)
                writer.add_scalar('train acc top-5', top5[-1].val, i*epoch+base_step)

                for name, param in model.named_parameters():
                    if 'bn' not in name:
                        writer.add_histogram(name, param, i*epoch+base_step)

                logging.info('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@3 {top3.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(train_loader_known),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_known),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg, running_lr





def train_known_unknown_in_order(train_loader_known,
                                train_loader_unknown,
                                model,
                                criterion,
                                optimizer,
                                epoch,
                                penalty_factors_known,
                                penalty_factors_unknown):

    ##########################################
    # Set up evaluation metrics
    ##########################################
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Initialize the accuracy for known and unknown respectively
    top1_known, top3_known, top5_known = [], [], []
    for i in range(args.nBlocks):
        top1_known.append(AverageMeter())
        top3_known.append(AverageMeter())
        top5_known.append(AverageMeter())

    top1_unknown, top3_unknown, top5_unknown = [], [], []
    for i in range(args.nBlocks):
        top1_unknown.append(AverageMeter())
        top3_unknown.append(AverageMeter())
        top5_unknown.append(AverageMeter())

    model.train()
    end = time.time()

    running_lr = None


    ###################################################
    # training known data: no RT, just normal training
    ###################################################
    save_known_txt_path = os.path.join(args.save, "train_known_stats_epoch_" + str(epoch) + ".txt")

    with open(save_known_txt_path, 'w') as train_f:
        for i, batch in enumerate(train_loader_known):
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=len(train_loader_known), method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            loss = 0.0

            known_input = batch["imgs"]
            known_target = batch["labels"] - 1

            known_input_var = torch.autograd.Variable(known_input)
            known_target = known_target.cuda(async=True)
            known_target_var = torch.autograd.Variable(known_target).long()

            output_1 = model(known_input_var)

            if not isinstance(output_1, list):
                output_1 = [output_1]

            for j in range(len(output_1)):
                if args.use_5_weights:
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output_1[j] * penalty_factor
                else:
                    output_weighted = output_1[j]

                loss += criterion(output_weighted, known_target_var)

            losses.update(loss.item(), known_input.size(0))


            ##########################################
            # Evaluate model on known samples
            ##########################################
            for j in range(len(output_1)):
                prec1_known, prec3_known, prec5_known = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                top1_known[j].update(prec1_known.item(), known_input.size(0))
                top3_known[j].update(prec3_known.item(), known_input.size(0))
                top5_known[j].update(prec5_known.item(), known_input.size(0))

            ###################################################
            # BP
            ###################################################
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
                      'Acc known@1 {top1.val:.4f}\t'
                      'Acc known@3 {top3.val:.4f}\t'
                      'Acc known@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(train_loader_known),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc known@1 {top1.val:.4f}\t'
                              'Acc known@3 {top3.val:.4f}\t'
                              'Acc known@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_known),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))

    ###################################################
    # training unknown data: with RTs
    ###################################################
    save_unknown_txt_path = os.path.join(args.save, "train_unknown_stats_epoch_" + str(epoch) + ".txt")

    with open(save_unknown_txt_path, 'w') as train_f:
        for i, batch in enumerate(train_loader_unknown):
            data_time.update(time.time() - end)

            loss = 0.0

            unknown_input = batch["imgs"]
            unknown_target = batch["labels"] - 1
            rts = batch["rts"]

            unknown_input_var = torch.autograd.Variable(unknown_input)
            unknown_target = unknown_target.cuda(async=True)
            unknown_target_var = torch.autograd.Variable(unknown_target).long()

            output_unknown = model(unknown_input_var)

            if not isinstance(output_unknown, list):
                output_unknown = [output_unknown]

            for j in range(len(output_unknown)):
                if args.use_5_weights:
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output_unknown[j] * penalty_factor
                else:
                    output_weighted = output_unknown[j]

                if args.use_pp_loss:
                    scale_factor = get_pp_factor(rts[j])
                    loss += scale_factor * criterion(output_weighted, unknown_target_var)
                else:
                    loss += criterion(output_weighted, unknown_target_var)

                loss += criterion(output_weighted, unknown_target_var)

            losses.update(loss.item(), unknown_input.size(0))

            ##########################################
            # Evaluate model
            ##########################################
            for j in range(len(output_unknown)):
                prec1_unknown, prec3_unknown, prec5_unknown = accuracy(output_unknown[j].data, unknown_target_var, topk=(1, 3, 5))
                top1_unknown[j].update(prec1_unknown.item(), unknown_input.size(0))
                top3_unknown[j].update(prec3_unknown.item(), unknown_input.size(0))
                top5_unknown[j].update(prec5_unknown.item(), unknown_input.size(0))

            ###################################################
            # BP
            ###################################################
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
                             'Acc unknown@1 {top1.val:.4f}\t'
                             'Acc unknown@3 {top3.val:.4f}\t'
                             'Acc unknown@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_unknown),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1_unknown[-1], top3=top3_unknown[-1], top5=top5_unknown[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader_unknown),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1_unknown[-1], top3=top3_unknown[-1], top5=top5_unknown[-1]))

    return losses.avg, top1_known[-1].avg, top3_known[-1].avg, top5_known[-1].avg, \
           top1_unknown[-1].avg, top3_unknown[-1].avg, top5_unknown[-1].avg, running_lr




def valid_known_unknown_in_order(valid_loader_known,
                                 valid_loader_unknown,
                                 model,
                                 criterion,
                                 epoch,
                                 penalty_factors_known,
                                 penalty_factors_unknown,
                                 base_step):
    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion_known:
    :param criterion_unknown:
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

    # Initialize the accuracy for known and unknown respectively
    top1_known, top3_known, top5_known = [], [], []
    for i in range(args.nBlocks):
        top1_known.append(AverageMeter())
        top3_known.append(AverageMeter())
        top5_known.append(AverageMeter())

    top1_unknown, top3_unknown, top5_unknown = [], [], []
    for i in range(args.nBlocks):
        top1_unknown.append(AverageMeter())
        top3_unknown.append(AverageMeter())
        top5_unknown.append(AverageMeter())

    model.train()
    end = time.time()

    ###################################################
    # training process setup...
    ###################################################
    save_known_txt_path = os.path.join(args.save, "valid_known_stats_epoch_" + str(epoch) + ".txt")

    with torch.no_grad():
        with open(save_known_txt_path, 'w') as valid_known_f:
            for i, batch in enumerate(valid_loader_known):
                data_time.update(time.time() - end)

                loss = 0.0

                ###################################################
                # training known data: no RT, just normal training
                ###################################################
                # print("Training the first batch")
                known_input = batch["imgs"]
                known_target = batch["labels"] - 1

                known_input_var = torch.autograd.Variable(known_input)
                known_target = known_target.cuda(async=True)
                known_target_var = torch.autograd.Variable(known_target).long()

                output_1 = model(known_input_var)

                if not isinstance(output_1, list):
                    output_1 = [output_1]

                for j in range(len(output_1)):
                    if args.use_5_weights:
                        penalty_factor = penalty_factors_known[j]
                        output_weighted = output_1[j] * penalty_factor
                    else:
                        output_weighted = output_1[j]

                    loss += criterion(output_weighted, known_target_var)

                losses.update(loss.item(), known_input.size(0))


                ##########################################
                # Evaluate model
                ##########################################
                for j in range(len(output_1)):
                    prec1_known, prec3_known, prec5_known = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                    top1_known[j].update(prec1_known.item(), known_input.size(0))
                    top3_known[j].update(prec3_known.item(), known_input.size(0))
                    top5_known[j].update(prec5_known.item(), known_input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:

                    logging.info('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Acc known@1 {top1.val:.4f}\t'
                          'Acc known@3 {top3.val:.4f}\t'
                          'Acc known@5 {top5.val:.4f}\n'.format(
                            epoch, i + 1, len(valid_loader_known),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))

                    valid_known_f.write('Epoch: [{0}][{1}/{2}]\t'
                                  'Time {batch_time.avg:.3f}\t'
                                  'Data {data_time.avg:.3f}\t'
                                  'Loss {loss.val:.4f}\t'
                                  'Acc known@1 {top1.val:.4f}\t'
                                  'Acc known@3 {top3.val:.4f}\t'
                                  'Acc known@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(valid_loader_known),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1_known[-1], top3=top3_known[-1], top5=top5_known[-1]))


    ###################################################
    # validate unknown
    ###################################################
    with torch.no_grad():
        save_unknown_txt_path = os.path.join(args.save, "valid_unknown_stats_epoch_" + str(epoch) + ".txt")

        with open(save_unknown_txt_path, 'w') as valid_unknown_f:
            for i, batch in enumerate(valid_loader_unknown):
                data_time.update(time.time() - end)

                loss = 0.0

                unknown_input = batch["imgs"]
                unknown_target = batch["labels"] - 1
                rts = batch["rts"]

                unknown_input_var = torch.autograd.Variable(unknown_input)
                unknown_target = unknown_target.cuda(async=True)
                unknown_target_var = torch.autograd.Variable(unknown_target).long()

                output_unknown = model(unknown_input_var)

                if not isinstance(output_unknown, list):
                    output_unknown = [output_unknown]

                for j in range(len(output_unknown)):
                    if args.use_5_weights:
                        penalty_factor = penalty_factors_unknown[j]
                        output_weighted = output_unknown[j] * penalty_factor
                    else:
                        output_weighted = output_unknown[j]

                    if args.use_pp_loss:
                        scale_factor = get_pp_factor(rts[j])
                        loss += scale_factor * criterion(output_weighted, unknown_target_var)
                    else:
                        loss += criterion(output_weighted, unknown_target_var)

                    loss += criterion(output_weighted, unknown_target_var)

                losses.update(loss.item(), unknown_input.size(0))

                ##########################################
                # Evaluate model
                ##########################################
                for j in range(len(output_unknown)):
                    prec1_unknown, prec3_unknown, prec5_unknown = accuracy(output_unknown[j].data, unknown_target_var, topk=(1, 3, 5))
                    top1_unknown[j].update(prec1_unknown.item(), unknown_input.size(0))
                    top3_unknown[j].update(prec3_unknown.item(), unknown_input.size(0))
                    top5_unknown[j].update(prec5_unknown.item(), unknown_input.size(0))

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    if i % args.print_freq == 0:
                        logging.info('Epoch: [{0}][{1}/{2}]\t'
                                     'Time {batch_time.avg:.3f}\t'
                                     'Data {data_time.avg:.3f}\t'
                                     'Loss {loss.val:.4f}\t'
                                     'Acc unknown@1 {top1.val:.4f}\t'
                                     'Acc unknown@3 {top3.val:.4f}\t'
                                     'Acc unknown@5 {top5.val:.4f}\n'.format(
                            epoch, i + 1, len(valid_loader_known),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1_unknown[-1], top3=top3_unknown[-1], top5=top5_unknown[-1]))

                        valid_unknown_f.write('Epoch: [{0}][{1}/{2}]\t'
                                      'Time {batch_time.avg:.3f}\t'
                                      'Data {data_time.avg:.3f}\t'
                                      'Loss {loss.val:.4f}\t'
                                      'Acc unknown@1 {top1.val:.4f}\t'
                                      'Acc unknown@3 {top3.val:.4f}\t'
                                      'Acc unknown@5 {top5.val:.4f}\n'.format(
                            epoch, i + 1, len(valid_loader_known),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1_unknown[-1], top3=top3_unknown[-1], top5=top5_unknown[-1]))

    return losses.avg, top1_known[-1].avg, top3_known[-1].avg, top5_known[-1].avg,\
           top1_unknown[-1].avg, top3_unknown[-1].avg, top5_unknown[-1].avg





def train_early_exit_with_pp_loss(train_loader_known,
                                  train_loader_unknown,
                                  model,
                                  criterion,
                                  optimizer,
                                  epoch,
                                  penalty_factors_known,
                                  penalty_factors_unknown,
                                  base_step,
                                  rt_max=20):
    """

    :param train_loader_known:
    :param train_loader_unknown:
    :param model:
    :param criterion_known:
    :param criterion_unknown:
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

    nb_known_batches = len(train_loader_known)
    nb_unknown_batches = len(train_loader_unknown)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)


    # Check which category has more samples/batches
    if nb_known_batches >= nb_unknown_batches:
        long_loader = train_loader_known
        short_loader = iter(train_loader_unknown)
    else:
        long_loader = train_loader_unknown
        short_loader = iter(train_loader_known)

    # TODO: Get batch for known and unknown at the same time
    with open(save_txt_path, 'w') as train_f:
        for i, batch_1 in enumerate(long_loader):
            try:
                batch_2 = next(short_loader)
            except StopIteration:
                batch_2 = next(iter(long_loader))

            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=nb_total_batches, method=args.lr_type)
            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            ###################################################
            # Different cases
            ###################################################
            cate_1 = batch_1["category"]
            cate_2 = batch_2["category"]

            # TODO: Case 1: one batch from known_known, another batch from known_unknown
            if cate_1 == "known_known" and cate_2 == "known_unknown":
                # print("case 1-1")
                loss = 0.0

                ###################################################
                # training known data: no RT, just normal training
                ###################################################
                # print("Training the first batch")
                known_input = batch_1["imgs"]
                known_target = batch_1["labels"] - 1

                known_input_var = torch.autograd.Variable(known_input)
                known_target = known_target.cuda(async=True)
                known_target_var = torch.autograd.Variable(known_target).long()

                output_1 = model(known_input_var)

                if not isinstance(output_1, list):
                    output_1 = [output_1]

                for j in range(len(output_1)):
                    if args.use_5_weights:
                        penalty_factor = penalty_factors_known[j]
                        output_weighted = output_1[j] * penalty_factor
                    else:
                        output_weighted = output_1[j]

                    loss += criterion(output_weighted, known_target_var)

                losses.update(loss.item(), known_input.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for j in range(len(output_1)):
                    prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                    top1[j].update(prec1.item(), known_input.size(0))
                    top3[j].update(prec3.item(), known_input.size(0))
                    top5[j].update(prec5.item(), known_input.size(0))


                ###################################################
                # Training unknown data: with RT + pp-loss
                ###################################################
                # print("Training the second batch")
                unknown_input = batch_2["imgs"]
                unknown_target = batch_2["labels"] - 1
                rts = batch_2["rts"]

                unknown_input_var = torch.autograd.Variable(unknown_input)
                unknown_target = unknown_target.cuda(async=True)
                unknown_target_var = torch.autograd.Variable(unknown_target).long()

                output_2 = model(unknown_input_var)

                if not isinstance(output_2, list):
                    output_2 = [output_2]

                for j in range(len(output_2)):
                    if args.use_5_weights:
                        penalty_factor = penalty_factors_unknown[j]
                        output_weighted = output_2[j] * penalty_factor
                    else:
                        output_weighted = output_2[j]

                    if args.use_pp_loss:
                        scale_factor = get_pp_factor(rts[j])
                        loss += scale_factor * criterion(output_weighted, unknown_target_var)
                    else:
                        loss += criterion(output_weighted, unknown_target_var)

                losses.update(loss.item(), unknown_input.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                for j in range(len(output_2)):
                    prec1, prec3, prec5 = accuracy(output_2[j].data, unknown_target_var, topk=(1, 3, 5))
                    top1[j].update(prec1.item(), unknown_input.size(0))
                    top3[j].update(prec3.item(), unknown_input.size(0))
                    top5[j].update(prec5.item(), unknown_input.size(0))

            else:
                print("something is wrong...")
                return

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
                    epoch, i + 1, len(long_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.avg:.3f}\t'
                              'Data {data_time.avg:.3f}\t'
                              'Loss {loss.val:.4f}\t'
                              'Acc@1 {top1.val:.4f}\t'
                              'Acc@3 {top3.val:.4f}\t'
                              'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(long_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))


    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg, running_lr





def validate_early_exit_with_pp_loss(val_known_loader,
                                     val_unknown_loader,
                                     model,
                                     criterion,
                                     penalty_factors_known,
                                     penalty_factors_unknown,
                                     rt_max=20,
                                     epoch=None):
    """

    :param val_known_loader:
    :param val_unknown_loader:
    :param model:
    :param criterion:
    :param penalty_factors_known:
    :param penalty_factors_unknown:
    :param rt_max:
    :param epoch:
    :return:
    """
    ##########################################
    # Set up evaluation metrics
    ##########################################
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()
    end = time.time()

    ##########################################
    # Validation process
    ##########################################
    # if valid_unknown == False:
    #     save_txt_path = os.path.join(args.save, "valid_known_stats_epoch_" + str(epoch) + ".txt")
    # else:
    save_txt_path = os.path.join(args.save, "valid_stats_epoch_" + str(epoch) + ".txt")

    nb_known_batches = len(val_known_loader)
    nb_unknown_batches = len(val_unknown_loader)
    nb_total_batches = nb_known_batches + nb_unknown_batches

    print("There are %d batches in known_known loader" % nb_known_batches)
    print("There are %d batches in known_unknown loader" % nb_unknown_batches)

    # Check which category has more samples/batches
    if nb_known_batches >= nb_unknown_batches:
        long_loader = val_known_loader
        short_loader = iter(val_unknown_loader)
    else:
        long_loader = val_unknown_loader
        short_loader = iter(val_known_loader)

    with torch.no_grad():
        with open(save_txt_path, 'w') as valid_f:
            for i, batch_1 in enumerate(long_loader):
                try:
                    batch_2 = next(short_loader)
                except StopIteration:
                    batch_2 = next(iter(long_loader))

                    ###################################################
                    # Different cases
                    ###################################################
                    cate_1 = batch_1["category"]
                    cate_2 = batch_2["category"]


                    # TODO: Case 1: one batch from known_known, another batch from known_unknown
                    if cate_1 == "known_known" and cate_2 == "known_unknown":
                        # print("case 1-1")
                        loss = 0.0

                        ###################################################
                        # training known data: no RT, just normal training
                        ###################################################
                        # print("Training the first batch")
                        known_input = batch_1["imgs"]
                        known_target = batch_1["labels"] - 1

                        known_input_var = torch.autograd.Variable(known_input)
                        known_target = known_target.cuda(async=True)
                        known_target_var = torch.autograd.Variable(known_target).long()

                        output_1 = model(known_input_var)

                        if not isinstance(output_1, list):
                            output_1 = [output_1]

                        for j in range(len(output_1)):
                            penalty_factor = penalty_factors_known[j]
                            output_weighted = output_1[j] * penalty_factor

                            loss += criterion(output_weighted, known_target_var)

                        losses.update(loss.item(), known_input.size(0))

                        ###################################################
                        # Training unknown data: with RT + pp-loss
                        ###################################################
                        # print("Training the second batch")
                        unknown_input = batch_2["imgs"]
                        unknown_target = batch_2["labels"] - 1
                        rts = batch_2["rts"]

                        unknown_input_var = torch.autograd.Variable(unknown_input)
                        unknown_target = unknown_target.cuda(async=True)
                        unknown_target_var = torch.autograd.Variable(unknown_target).long()

                        output_2 = model(unknown_input_var)

                        if not isinstance(output_2, list):
                            output_2 = [output_2]

                        for j in range(len(output_2)):
                            penalty_factor = penalty_factors_unknown[j]
                            output_weighted = output_2[j] * penalty_factor

                            scale_factor = get_pp_factor(rts[j])
                            loss += scale_factor * criterion(output_weighted, unknown_target_var)

                        losses.update(loss.item(), unknown_input.size(0))

                        ##########################################
                        # Evaluate model
                        ##########################################
                        for j in range(len(output_1)):
                            prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), known_input.size(0))
                            top3[j].update(prec3.item(), known_input.size(0))
                            top5[j].update(prec5.item(), known_input.size(0))

                        for j in range(len(output_2)):
                            prec1, prec3, prec5 = accuracy(output_2[j].data, unknown_target_var, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), unknown_input.size(0))
                            top3[j].update(prec3.item(), unknown_input.size(0))
                            top5[j].update(prec5.item(), unknown_input.size(0))


                    elif cate_2 == "known_known" and cate_1 == "known_unknown":
                        # print("case 1-2")
                        loss = 0.0
                        ###################################################
                        # training known data: no RT, just normal training
                        ###################################################
                        # print("Training the first batch")
                        known_input = batch_2["imgs"]
                        known_target = batch_2["labels"] - 1

                        known_input_var = torch.autograd.Variable(known_input)
                        known_target = known_target.cuda(async=True)
                        known_target_var = torch.autograd.Variable(known_target).long()

                        output_1 = model(known_input_var)
                        # print(len(output_1))

                        if not isinstance(output_1, list):
                            output_1 = [output_1]

                        for j in range(len(output_1)):
                            penalty_factor = penalty_factors_known[j]
                            output_weighted = output_1[j] * penalty_factor

                            loss += criterion(output_weighted, known_target_var)

                        losses.update(loss.item(), known_input.size(0))

                        ###################################################
                        # Training unknown data: with RT + pp-loss
                        ###################################################
                        # print("Training the second batch")
                        unknown_input = batch_1["imgs"]
                        unknown_target = batch_1["labels"] - 1
                        rts = batch_1["rts"]

                        unknown_input_var = torch.autograd.Variable(unknown_input)
                        unknown_target = unknown_target.cuda(async=True)
                        unknown_target_var = torch.autograd.Variable(unknown_target).long()

                        output_2 = model(unknown_input_var)
                        # print(len(output_2))

                        if not isinstance(output_2, list):
                            output_2 = [output_2]

                        for j in range(len(output_2)):
                            penalty_factor = penalty_factors_unknown[j]
                            output_weighted = output_2[j] * penalty_factor

                            scale_factor = get_pp_factor(rts[j])
                            loss += scale_factor * criterion(output_weighted, unknown_target_var)

                        losses.update(loss.item(), unknown_input.size(0))

                        ##########################################
                        # Evaluate model
                        ##########################################
                        for j in range(len(output_1)):
                            prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), known_input.size(0))
                            top3[j].update(prec3.item(), known_input.size(0))
                            top5[j].update(prec5.item(), known_input.size(0))

                        for j in range(len(output_2)):
                            prec1, prec3, prec5 = accuracy(output_2[j].data, unknown_target_var, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), unknown_input.size(0))
                            top3[j].update(prec3.item(), unknown_input.size(0))
                            top5[j].update(prec5.item(), unknown_input.size(0))



                    # TODO: Case 2: both batches from known_known
                    elif cate_1 == "known_known" and cate_2 == "known_known":
                        # print("case 2")
                        loss = 0.0

                        ###################################################
                        # Training the first known batch
                        ###################################################
                        # print("Training the first batch")
                        known_input_1 = batch_1["imgs"]
                        known_target_1 = batch_1["labels"] - 1

                        known_input_var_1 = torch.autograd.Variable(known_input_1)
                        known_target_1 = known_target_1.cuda(async=True)
                        known_target_var_1 = torch.autograd.Variable(known_target_1).long()

                        output_1 = model(known_input_var_1)

                        if not isinstance(output_1, list):
                            output_1 = [output_1]

                        for j in range(len(output_1)):
                            penalty_factor = penalty_factors_known[j]
                            output_weighted = output_1[j] * penalty_factor

                            loss += criterion(output_weighted, known_target_var_1)

                        losses.update(loss.item(), known_input_1.size(0))

                        ###################################################
                        # Training the 2nd known batch
                        ###################################################
                        # print("Training the second batch")
                        known_input_2 = batch_2["imgs"]
                        known_target_2 = batch_2["labels"] - 1

                        known_input_var_2 = torch.autograd.Variable(known_input_2)
                        known_target_2 = known_target_2.cuda(async=True)
                        known_target_var_2 = torch.autograd.Variable(known_target_2).long()

                        output_2 = model(known_input_var_2)

                        if not isinstance(output_2, list):
                            output_2 = [output_2]

                        for j in range(len(output_2)):
                            penalty_factor = penalty_factors_known[j]
                            output_weighted = output_2[j] * penalty_factor

                            loss += criterion(output_weighted, known_target_var_2)

                        losses.update(loss.item(), known_input_2.size(0))

                        ##########################################
                        # Evaluate model
                        ##########################################
                        # try:
                        for j in range(len(output_1)):
                            prec1, prec3, prec5 = accuracy(output_1[j].data, known_target_var_1, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), known_input_1.size(0))
                            top3[j].update(prec3.item(), known_input_1.size(0))
                            top5[j].update(prec5.item(), known_input_1.size(0))

                        for j in range(len(output_2)):
                            prec1, prec3, prec5 = accuracy(output_2[j].data, known_target_var_2, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), known_input_2.size(0))
                            top3[j].update(prec3.item(), known_input_2.size(0))
                            top5[j].update(prec5.item(), known_input_2.size(0))



                    # TODO: Case 3: both batches from known_unknown
                    elif cate_1 == "known_unknown" and cate_2 == "known_unknown":
                        # print("case 3")
                        loss = 0.0

                        ###################################################
                        # training the 1st batch of unknown
                        ###################################################
                        # print("Training the first batch")
                        unknown_input_1 = batch_1["imgs"]
                        unknown_target_1 = batch_1["labels"] - 1
                        rts_1 = batch_1["rts"]

                        unknown_input_var_1 = torch.autograd.Variable(unknown_input_1)
                        unknown_target_1 = unknown_target_1.cuda(async=True)
                        unknown_target_var_1 = torch.autograd.Variable(unknown_target_1).long()

                        output_1 = model(unknown_input_var_1)

                        if not isinstance(output_1, list):
                            output_1 = [output_1]

                        for j in range(len(output_1)):
                            penalty_factor = penalty_factors_unknown[j]
                            output_weighted = output_1[j] * penalty_factor

                            scale_factor = rt_max - rts_1[j]
                            loss += scale_factor * criterion(output_weighted, unknown_target_var_1)

                        losses.update(loss.item(), unknown_input.size(0))

                        ###################################################
                        # training the 2nd batch of unknown
                        ###################################################
                        # print("Training the second batch")
                        unknown_input_2 = batch_2["imgs"]
                        unknown_target_2 = batch_2["labels"] - 1
                        rts_2 = batch_2["rts"]

                        unknown_input_var_2 = torch.autograd.Variable(unknown_input_2)
                        unknown_target_2 = unknown_target_2.cuda(async=True)
                        unknown_target_var_2 = torch.autograd.Variable(unknown_target_2).long()

                        output_2 = model(unknown_input_var_2)

                        if not isinstance(output_2, list):
                            output_2 = [output_2]

                        for j in range(len(output_2)):
                            penalty_factor = penalty_factors_unknown[j]
                            output_weighted = output_2[j] * penalty_factor

                            scale_factor = rt_max - rts_2[j]
                            loss += scale_factor * criterion(output_weighted, unknown_target_var_2)

                        losses.update(loss.item(), unknown_input.size(0))

                        ##########################################
                        # Evaluate model
                        ##########################################
                        for j in range(len(output_1)):
                            prec1, prec3, prec5 = accuracy(output_1[j].data, unknown_target_var_1, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), unknown_input_1.size(0))
                            top3[j].update(prec3.item(), unknown_input_1.size(0))
                            top5[j].update(prec5.item(), unknown_input_1.size(0))

                        for j in range(len(output_2)):
                            prec1, prec3, prec5 = accuracy(output_2[j].data, unknown_target_var_2, topk=(1, 3, 5))
                            top1[j].update(prec1.item(), unknown_input_2.size(0))
                            top3[j].update(prec3.item(), unknown_input_2.size(0))
                            top5[j].update(prec5.item(), unknown_input_2.size(0))

                    # TODO: Case 4: something is wrong...
                    else:
                        print("something is wrong...")
                        return


                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    logging.info('Epoch: [{0}/{1}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Acc@1 {top1.val:.4f}\t'
                          'Acc@3 {top3.val:.4f}\t'
                          'Acc@5 {top5.val:.4f}'.format(
                            i + 1, nb_total_batches,
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                    valid_f.write('Epoch: [{0}][{1}/{2}]\t'
                                  'Time {batch_time.avg:.3f}\t'
                                  'Data {data_time.avg:.3f}\t'
                                  'Loss {loss.val:.4f}\t'
                                  'Acc@1 {top1.val:.4f}\t'
                                  'Acc@3 {top3.val:.4f}\t'
                                  'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, nb_total_batches,
                        batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    for j in range(args.nBlocks):
        logging.info(' * Validation accuracy: top-1:{top1.avg:.3f} top-3:{top3.avg:.3f} top-5:{top5.avg:.3f}'.format(top1=top1[j], top3=top3[j], top5=top5[j]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg



def test_with_novelty_pp(data_loader,
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
        for i, batch in enumerate(data_loader):
            print("*" * 50)

            input = batch["imgs"]
            target = batch["labels"] - 1

            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            rts = []


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
            # target_var = torch.autograd.Variable(target)
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

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
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg


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
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
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
