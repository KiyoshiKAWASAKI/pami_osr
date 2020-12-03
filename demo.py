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
import sys
import warnings
warnings.filterwarnings("ignore")
import random
from args import arg_parser
import torch.nn as nn
import models

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)


###############################################
# Change these parameters
###############################################
model_name = "msd_net"

use_json_data = True
use_5_weights = False
use_pp_loss = False
run_test = False

n_epochs = 100
batch_size = 32
thresh_top_1 = 0.90
nb_training_classes = 336 # known:335, unknown:1
nBlocks = 5

penalty_factors_for_known = [1.0, 2.5, 5.0, 7.5, 10.0]
penalty_factors_for_novel = [3.897, 5.390, 7.420, 11.491, 22.423]

model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/dense_net/1117_base_setup"

train_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
train_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
valid_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
valid_known_unknown_path =  "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
test_known_known_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_known_50.json"
test_known_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"
test_unknown_unknown_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/npy_json_files/debug_known_unknown_50.json"

# The folder path for saving everything
save_path_base = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on"

save_path_sub = "combo_pipeline/1203/msd_net_no_pp"
save_path = save_path_base + "/" + save_path_sub



def train_valid_one_epoch(known_loader,
                         unknown_loader,
                        model,
                        criterion,
                        optimizer,
                        nb_epoch,
                        penalty_factors_known,
                        penalty_factors_unknown,
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

    if use_msd_net:
        top1, top3, top5 = [], [], []
        for i in range(nBlocks):
            top1.append(AverageMeter())
            top3.append(AverageMeter())
            top5.append(AverageMeter())
    else:
        top1 = AverageMeter()
        top3 = AverageMeter()
        top5 = AverageMeter()

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
                batch_type = "known"

            elif i in unknown_indices:
                batch = next(unknown_iter)
                batch_type = "unknown"

            input = batch["imgs"]
            target = batch["labels"] - 1
            rts = batch["rts"]

            input_var = torch.autograd.Variable(input).cuda()
            target = target.cuda(async=True)
            target_var = torch.autograd.Variable(target).long()

            output = model(input_var)

            if not isinstance(output, list):
                output = [output]

            # Case 1: known batch + 5 weights
            if (batch_type == "known") and (use_5_weights == True):
                for j in range(len(output)):
                    penalty_factor = penalty_factors_known[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 2: known batch + no 5 weights
            if (batch_type == "known") and (use_5_weights == False):
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 3: unknown batch + no 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == False) and (use_pp_loss == False):
                for j in range(len(output)):
                    output_weighted = output[j]
                    loss += criterion(output_weighted, target_var)

            # Case 4: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == False):
                for j in range(len(output)):
                    penalty_factor = penalty_factors_unknown[j]
                    output_weighted = output[j] * penalty_factor
                    loss += criterion(output_weighted, target_var)

            # Case 5: unknown batch + 5 weights + no pp loss
            if (batch_type == "unknown") and (use_5_weights == True) and (use_pp_loss == True):
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

            if train_phase:
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
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    # with open(os.path.join(save, 'results.csv'), 'w') as f:
    #     f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')


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
                                                                    penalty_factors_known=penalty_factors_for_known,
                                                                    penalty_factors_unknown=penalty_factors_for_novel,
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
                                                                  penalty_factors_known=penalty_factors_for_known,
                                                                  penalty_factors_unknown=penalty_factors_for_novel,
                                                                  train_phase=False)

        elif model_name == "dense_net":
            train_loss, train_acc_top1, \
            train_acc_top3, train_acc_top5 = train_valid_one_epoch(known_loader=train_known_known_loader,
                                                                   unknown_loader=train_known_unknown_loader,
                                                                   model=model_wrapper,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   nb_epoch=epoch,
                                                                   penalty_factors_known=penalty_factors_for_known,
                                                                   penalty_factors_unknown=penalty_factors_for_novel,
                                                                   use_msd_net=False,
                                                                   train_phase=True)

            scheduler.step()

            valid_loss, valid_acc_top1, \
            valid_acc_top3, valid_acc_top5 = train_valid_one_epoch(known_loader=valid_known_known_loader,
                                                                   unknown_loader=valid_known_unknown_loader,
                                                                   model=model_wrapper,
                                                                   criterion=criterion,
                                                                   optimizer=optimizer,
                                                                   nb_epoch=epoch,
                                                                   penalty_factors_known=penalty_factors_for_known,
                                                                   penalty_factors_unknown=penalty_factors_for_novel,
                                                                   use_msd_net=True,
                                                                   train_phase=False)

        else:
            return


        # Determine if model is the best
        if valid_loader:
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                print('New best top-1 accuracy: %.4f' % best_acc_top1)
                torch.save(model.state_dict(), save + "/model_epoch_" + str(epoch) + '.dat')
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
def test_with_novelty(test_loader,
                      model,
                      test_unknown):
    """

    :param val_loader:
    :param model:
    :param criterion:
    :return:
    """

    # Set the model to evaluation mode
    model.cuda()
    model.eval()

    # Define the softmax - do softmax to each block.
    sm = torch.nn.Softmax(dim=1)

    # full_original_label_list = []
    # full_prob_list = []
    # full_target_list = []
    # full_rt_list = []

    sample_count = 0
    total_rt_count = 0

    correct_count = 0
    wrong_count = 0


    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # print("*" * 50)

            # original_label = target
            # print("Correct label:")
            # print(original_label)

            sample_count += 1

            # rts = []
            input = input.cuda()
            target = target.cuda(async=True)

            # print("Correct label:")
            # print(target)

            # Save original labels to the list
            # original_label_list = np.array(target.cpu().tolist())
            # for label in original_label_list:
            #     full_original_label_list.append(label)

            # Check the target labels: keep or change
            if test_unknown:
                for k in range(len(target)):
                    target[k] = -1

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)


            # Get the model outputs and RTs
            # print("Timer started.")
            start =timer()
            output, end_time = model(input_var)

            rt = end_time[0]-start
            total_rt_count += rt

            # extract the probability and apply our threshold
            prob = sm(output)
            prob_list = np.array(prob.cpu().tolist())
            max_prob = np.max(prob_list)

            # decide whether to do classification or reject
            # When the probability is larger than our threshold
            if max_prob >= thresh_top_1:
                # print("Max top-1 probability is %f, larger than threshold %f" % (max_prob, thresh_top_1))

                pred_label = torch.argmax(output)
                # print("Predicted label:")
                # print(pred_label)

            # When the probability is smaller than our threshold
            else:
                pred_label = -1

            if pred_label == target:
                # print("Right prediction!")
                correct_count += 1
            else:
                # print("Wrong prediction!")
                wrong_count += 1

    print("Total number of Samples: %d" % sample_count)
    print("Number or right predictions: %d" % correct_count)
    print("Number of wrong predictions: %d" % wrong_count)

    avg_rt = total_rt_count / sample_count
    print("Average RT: % 4f" % avg_rt)

    acc = float(correct_count)/float(correct_count+wrong_count)
    print("TOP-1 accuracy: %4f" % acc)



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
    # Create model: MSD-Net or other
    ########################################################################
    if model_name == "dense_net":
        print("Training DenseNet")
        model = efficient_dense_net.DenseNet(growth_rate=growth_rate,
                                            block_config=block_config,
                                            num_init_features=growth_rate * 2,
                                            num_classes=335,
                                            small_inputs=True,
                                            efficient=efficient)

    # TODO: Add creating MSD-Net here
    elif model_name == "msd_net":
        model = getattr(models, args.arch)(args)



    # TODO: Maybe adding other networks in the future
    else:
        pass



    ########################################################################
    # Test-only or Training + validation
    ########################################################################
    # TODO: Fix this testing process + add op count
    if run_test:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.dat')))

        print("*" * 50)
        print("Testing the known samples...")
        test_with_novelty(test_loader=test_known_known_loader,
                          model=model,
                          test_unknown=False)

        print("*" * 50)
        print("testing the unknown samples...")
        test_with_novelty(test_loader=test_unknown_unknown_loader,
                          model=model,
                          test_unknown=True)
        print("*" * 50)

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