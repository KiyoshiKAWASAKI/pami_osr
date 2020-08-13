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
import csv
import logging

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from timeit import default_timer as timer
import datetime


args = arg_parser.parse_args()
torch.manual_seed(args.seed)



if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

log_file_path = args.log_file_path

logging.basicConfig(filename=log_file_path,
                    level=logging.DEBUG)
# logging.basicConfig(stream=sys.stdout,
#                     level=logging.DEBUG)
logging_header = "date: {}".format(datetime.datetime.now())



class train_msdnet():
    def __init__(self):
        self.args = arg_parser.parse_args()


def main():
    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = getattr(models, args.arch)(args, nb_blocks=5)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

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

    train_loader, val_loader, test_loader = get_dataloaders(args)

    # This is for test only
    if args.evalmode is not None:
        logging.info("Doing testing only.")
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)

        if args.evalmode == 'anytime':
            if args.train_k_plus_1 or args.test_with_novel:
                test_with_novelty(val_loader=test_loader,
                                  model=model,
                                  criterion=criterion)
            else:
                validate(test_loader, model, criterion)

        else:
            logging.info("Only supporting anytime prediction!")

        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1\tval_prec1\ttrain_prec5\tval_prec5']

    # Here is for training and validation
    for epoch in range(args.start_epoch, args.epochs):
        # Adding the option for training k+1 classes
        if args.train_k_plus_1:
            logging.info("Training MSD-Net on K+1 classes.")

            train_loss, train_prec1, train_prec3, train_prec5, lr = train_k_plus_one(train_loader, model, criterion, optimizer, epoch)
            val_loss, val_prec1, val_prec3, val_prec5 = validate_k_plus_one(val_loader, model, criterion, epoch)

        # Adding the option for training early exits using diff penalties.
        elif args.train_early_exit:
            logging.info("Training with weighted loss for different classes.")
            # TODO: define the penalty factors here
            penalty_factors = [0.2, 0.4, 0.6, 0.8, 1.0]

            train_loss, train_prec1,train_prec5, lr = train_early_exit_loss(train_loader,
                                                                            model,
                                                                            criterion,
                                                                            optimizer,
                                                                            epoch,
                                                                            penalty_factors,
                                                                            "simple")

            # TODO: Which validate function should we use? Probably testing_with_novelty?
            # TODO: need to update original validate to top 1,3,5
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch)


        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6).format(epoch, lr, train_loss, val_loss, train_prec1, val_prec1, train_prec5, val_prec5))

        # Find the best model
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




def train():
    pass




def train_early_exit_loss(train_loader,
                          model,
                          criterion,
                          optimizer,
                          epoch,
                          penalty_factors,
                          strategy,
                          nb_known_classes=325,
                          nb_known_unknown_classes=44,
                          nb_unknown_classes=44):
    """
    Modify how the loss is calculated:
    assign smaller penalties to earlier exits, and larger penalties to later exits.

    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :return:
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # TODO: Update the evaluation metrics to top-1, top-3 and top-5
    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    running_lr = None

    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        input_var = torch.autograd.Variable(input)

        if strategy == "simple":
            # Implement the simple weighted loss strategy
            # by multiplying n weights to n exits respectively

            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            loss = 0.0

            # Just add the penalty factors here
            for j in range(len(output)):
                # print(output[j].shape) # Shape: [batch, nb_classes]
                # Assign different weights to the losses
                penalty_factor = penalty_factors[j]
                output_weighted = output[j] * penalty_factor

                loss += criterion(output_weighted, target_var)

            losses.update(loss.item(), input.size(0))

        # TODO: Implement the more complex strategies
        elif strategy == "complex":
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))



        for j in range(len(output)):
            prec1, prec3, prec5 = accuracy(output[j].data, target, topk=(1, 3, 5))
            top1[j].update(prec1.item(), input.size(0))
            top3[j].update(prec3.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}\n'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr




def train_k_plus_one(train_loader, model, criterion, optimizer, epoch):
    """
    Training k known classes and all other classes as +1 unknown class.

    :param train_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epoch:
    :return:
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # Update the evaluation metrics to top-1, top-3 and top-5
    top1, top3, top5 = [], [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top3.append(AverageMeter())
        top5.append(AverageMeter())

    model.train()

    end = time.time()

    running_lr = None

    with open(os.path.join(args.save, "training_stats_epoch_" + str(epoch) + ".txt"), 'w') as train_f:
        for i, (input, target) in enumerate(train_loader):
            lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                      nBatch=len(train_loader), method=args.lr_type)

            if running_lr is None:
                running_lr = lr

            data_time.update(time.time() - end)

            target = target.cuda(async=True)

            # Check the labels, change the label if it belongs to "unknown"
            # Be reminded that the label generated by data loader starts from 0
            for k in range(len(target)):
                if target[k] >= args.nb_training_classes - 1:
                    target[k] = args.nb_training_classes - 1

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec3, prec5 = accuracy(output[j].data, target, topk=(1, 3, 5))
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
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@3 {top3.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        epoch, i + 1, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

                train_f.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@3 {top3.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(train_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg, running_lr





def validate_k_plus_one(val_loader, model, criterion, epoch):
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
    with torch.no_grad():
        with open(os.path.join(args.save, "validation_stats_epoch_" + str(epoch) + ".txt"), 'w') as valid_f:
            for i, (input, target) in enumerate(val_loader):
                target = target.cuda(async=True)
                input = input.cuda()

                # Check the labels, change the label if it belongs to "unknown"
                # Be reminded that the label generated by data loader starts from 0
                for k in range(len(target)):
                    if target[k] >= args.nb_training_classes - 1:
                        target[k] = args.nb_training_classes - 1

                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                data_time.update(time.time() - end)

                output = model(input_var)
                if not isinstance(output, list):
                    output = [output]

                loss = 0.0
                for j in range(len(output)):
                    loss += criterion(output[j], target_var)

                losses.update(loss.item(), input.size(0))

                for j in range(len(output)):
                    prec1, prec3, prec5 = accuracy(output[j].data, target, topk=(1, 3, 5))
                    top1[j].update(prec1.item(), input.size(0))
                    top3[j].update(prec3.item(), input.size(0))
                    top5[j].update(prec5.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Epoch: [{0}/{1}]\t'
                          'Time {batch_time.avg:.3f}\t'
                          'Data {data_time.avg:.3f}\t'
                          'Loss {loss.val:.4f}\t'
                          'Acc@1 {top1.val:.4f}\t'
                          'Acc@3 {top1.val:.4f}\t'
                          'Acc@5 {top5.val:.4f}'.format(
                            i + 1, len(val_loader),
                            batch_time=batch_time, data_time=data_time,
                            loss=losses, top1=top1[-1], top3=top3[-1],top5=top5[-1]))

                    valid_f.write('Epoch: [{0}][{1}/{2}]\t'
                                  'Time {batch_time.avg:.3f}\t'
                                  'Data {data_time.avg:.3f}\t'
                                  'Loss {loss.val:.4f}\t'
                                  'Acc@1 {top1.val:.4f}\t'
                                  'Acc@3 {top3.val:.4f}\t'
                                  'Acc@5 {top5.val:.4f}\n'.format(
                        epoch, i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))


        for j in range(args.nBlocks):
            print(' * prec@1 {top1.avg:.3f} prec@3 {top3.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j],
                                                                                                top3=top3[j],
                                                                                            top5=top5[j]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg




def validate(val_loader, model, criterion, epoch=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input = input.cuda()

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg




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
            if not args.test_with_novel:
                loss = 0.0
                for j in range(len(output)):
                    loss += criterion(output[j], target_var)

                losses.update(loss.item(), input.size(0))



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
    print("=> saving checkpoint '{}'".format(model_filename))

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
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
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
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print("Here is the pred in accuracy function")
    # print(pred)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # print("Here is the output for correct:")
    # print(correct)

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




def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

if __name__ == '__main__':
    main()
