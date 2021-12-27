import os
import time
import torch
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
from tqdm import tqdm

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
use_performance_loss = False
use_exit_loss = False
# thresh = 0.7
# cross_entropy_weight = 1.0
# perform_loss_weight = 0.0
# exit_loss_weight = 1.0
random_seed = 0

###################################################################
# Training options #
###################################################################
model_name = "msd_net"

debug = False
use_pre_train = False
train_binary = False
use_new_loader = False
use_addition = True

###################################################################
# Test process options #
###################################################################
nb_itrs = 1000
use_trained_weights = True
run_one_sample = False


###################################################################
# Paths for saving results #
###################################################################
save_path_sub = "cross_entropy_known_only"


####################################################################
# Normally, there is no need to change these #
####################################################################
use_json_data = True
save_training_prob = False

nb_itr = 30
nb_clfs = 5
img_size = 224
nBlocks = 5
nb_classes = 294
rt_max = 28

if debug:
    n_epochs = 3
else:
    n_epochs = 200


batch_size = 16

if train_binary:
    nb_training_classes = 2
else:
    nb_training_classes = 294



#########################################################################################
# Define paths for saving model and data source #
#########################################################################################
save_path_base = "/scratch365/jhuang24/sail-on/models/msd_net"
save_path_with_date = save_path_base + "/" + date

if not save_path_with_date:
    os.mkdir(save_path_with_date)

if debug:
    save_path = save_path_with_date + "/debug_" + save_path_sub + "/seed_" + str(random_seed)
else:
    save_path = save_path_with_date + "/" + save_path_sub + "/seed_" + str(random_seed)

if debug:
    pass
else:
    if use_new_loader == False:
        train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                 "dataset_v1_3_partition/npy_json_files_shuffled/train_known_known.json"
        train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                   "dataset_v1_3_partition/npy_json_files_shuffled/train_known_unknown.json"

        valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                 "dataset_v1_3_partition/npy_json_files_shuffled/valid_known_known.json"
        valid_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                   "dataset_v1_3_partition/npy_json_files_shuffled/valid_known_unknown.json"
    else:
        train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                 "npy_json_files/rt_group_json/train_known_known.json"
        train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                   "npy_json_files/rt_group_json/train_known_unknown.json"
        valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                 "npy_json_files/rt_group_json/valid_known_known.json"
        valid_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/dataset_v1_3_partition/" \
                                   "npy_json_files/rt_group_json/valid_known_unknown.json"

    test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                            "dataset_v1_3_partition/npy_json_files_shuffled/test_known_known.json"
    test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                              "dataset_v1_3_partition/npy_json_files_shuffled/test_known_unknown.json"
    test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_51/open_set/data/" \
                                "dataset_v1_3_partition/npy_json_files_shuffled/test_unknown_unknown.json"


#########################################################################################
# Define all the functions #
#########################################################################################
def train_valid_test_one_epoch(known_loader,
                              model,
                              criterion,
                              optimizer,
                              nb_epoch,
                              use_msd_net,
                              phase):
    """

    :param known_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param nb_epoch:
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

    if phase == "train":
        model.train()
    else:
        model.eval()

    end = time.time()

    running_lr = None

    ###################################################
    # training process setup...
    ###################################################
    if phase == "train":
        save_txt_path = os.path.join(save_path, "train_stats_epoch_" + str(nb_epoch) + ".txt")
    elif phase == "valid":
        save_txt_path = os.path.join(save_path, "valid_stats_epoch_" + str(nb_epoch) + ".txt")
    else:
        save_txt_path = os.path.join(save_path, "test_stats_epoch_" + str(nb_epoch) + ".txt")

    # Count number of batches for known and unknown respectively
    nb_total_batches = len(known_loader)
    print("There are %d batches in known_known loader" % nb_total_batches)

    # Create iterator
    known_iter = iter(known_loader)

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
            batch = next(known_iter)

            input = batch["imgs"]
            target = batch["labels"]

            # Convert into PyTorch tensor
            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output, feature, end_time = model(input_var)

            if not isinstance(output, list):
                output = [output]

            ##########################################
            # Only MSD-Net
            ##########################################
            if model_name == "msd_net":
                for j in range(len(output)):
                    ce_loss = criterion(output[j], target_var)

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

            if phase == "train":
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                f.write('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.4f}\t'
                        'Acc@1 {top1.val:.4f}\t'
                        'Acc@3 {top3.val:.4f}\t'
                        'Acc@5 {top5.val:.4f}\n'.format(
                    nb_epoch, i + 1, nb_total_batches,
                    loss=losses, top1=top1[-1], top3=top3[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top3[-1].avg, top5[-1].avg



def train_valid_test(model,
                      train_known_known_loader,
                      valid_known_known_loader,
                      test_known_known_loader,
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

    # Optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

    # TODO: use pre-train model??
    if use_pre_train:
        pass

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch, '
                'train_loss, train_acc_top1, train_acc_top3, train_acc_top5, '
                'valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5,'
                'test_loss, test_acc_top1, test_acc_top3, test_acc_top5,\n')

    # Train model
    best_acc_top5 = 0.00

    for epoch in range(n_epochs):
        print("*" * 60)
        print("EPOCH:", epoch)
        if model_name == "msd_net":
            train_loss, train_acc_top1, \
            train_acc_top3, train_acc_top5 = train_valid_test_one_epoch(known_loader=train_known_known_loader,
                                                                   model=model_wrapper,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   nb_epoch=epoch,
                                                                   use_msd_net=True,
                                                                   phase="train")

            scheduler.step()

            valid_loss, valid_acc_top1, \
            valid_acc_top3, valid_acc_top5 = train_valid_test_one_epoch(known_loader=valid_known_known_loader,
                                                                   model=model_wrapper,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   nb_epoch=epoch,
                                                                   use_msd_net=True,
                                                                   phase="valid")

            test_loss, test_acc_top1, \
            test_acc_top3, test_acc_top5 = train_valid_test_one_epoch(known_loader=test_known_known_loader,
                                                                      model=model_wrapper,
                                                                      criterion=criterion,
                                                                      optimizer=optimizer,
                                                                      nb_epoch=epoch,
                                                                      use_msd_net=True,
                                                                      phase="test")

        else:
            pass

        # Determine if model is the best
        if valid_loader:
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
                print('New best top-5 validation accuracy: %.4f' % best_acc_top5)
            torch.save(model.state_dict(), save + "/model_epoch_" + str(epoch) + '.dat')
            torch.save(optimizer.state_dict(), save + "/optimizer_epoch_" + str(epoch) + '.dat')
        else:
            torch.save(model.state_dict(), save + "/model_epoch_" + str(epoch) + '.dat')

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d, '
                    '%0.6f, %0.6f, %0.6f, %0.6f, '
                    '%0.5f, %0.6f, %0.6f, %0.6f, '
                    '%0.5f, %0.6f, %0.6f, %0.6f,\n'% ((epoch + 1),
                                                       train_loss, train_acc_top1, train_acc_top3, train_acc_top5,
                                                       valid_loss, valid_acc_top1, valid_acc_top3, valid_acc_top5,
                                                       test_loss, test_acc_top1, test_acc_top3, test_acc_top5))



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
    # Use the new data loader here: no collate function??
    if use_new_loader:
        # Training
        train_known_known_dataset = msd_net_with_grouped_rts(json_path=train_known_known_path,
                                                             transform=train_transform)
        train_known_unknown_dataset = msd_net_with_grouped_rts(json_path=train_known_unknown_path,
                                                               transform=train_transform)

        train_known_known_index = torch.randperm(len(train_known_known_dataset))
        train_known_unknown_index = torch.randperm(len(train_known_unknown_dataset))

        train_known_known_loader = torch.utils.data.DataLoader(train_known_known_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               drop_last=True,
                                                               collate_fn=customized_dataloader.collate_new,
                                                               sampler=torch.utils.data.RandomSampler(
                                                                   train_known_known_index))
        train_known_unknown_loader = torch.utils.data.DataLoader(train_known_unknown_dataset,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 drop_last=True,
                                                                 collate_fn=customized_dataloader.collate_new,
                                                                 sampler=torch.utils.data.RandomSampler(
                                                                     train_known_unknown_index))

        # Validation
        valid_known_known_dataset = msd_net_with_grouped_rts(json_path=valid_known_known_path,
                                                             transform=valid_transform)
        valid_known_unknown_dataset = msd_net_with_grouped_rts(json_path=valid_known_unknown_path,
                                                               transform=valid_transform)

        valid_known_known_index = torch.randperm(len(valid_known_known_dataset))
        valid_known_unknown_index = torch.randperm(len(valid_known_unknown_dataset))

        valid_known_known_loader = torch.utils.data.DataLoader(valid_known_known_dataset,
                                                               batch_size=batch_size,
                                                               shuffle=False,
                                                               collate_fn=customized_dataloader.collate_new,
                                                               sampler=torch.utils.data.RandomSampler(
                                                                   valid_known_known_index))
        valid_known_unknown_loader = torch.utils.data.DataLoader(valid_known_unknown_dataset,
                                                                 batch_size=batch_size,
                                                                 shuffle=False,
                                                                 drop_last=True,
                                                                 collate_fn=customized_dataloader.collate_new,
                                                                 sampler=torch.utils.data.RandomSampler(
                                                                     valid_known_unknown_index))


    else:
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
        pass

    ########################################################################
    # Test-only or Training + validation
    ########################################################################

    # Make save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)

    # Combine training all networks together
    train_valid_test(model=model,
                     train_known_known_loader=train_known_known_loader,
                     valid_known_known_loader=valid_known_known_loader,
                     test_known_known_loader=test_known_known_loader,
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