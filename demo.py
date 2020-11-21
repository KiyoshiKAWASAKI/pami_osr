import fire
import os
import time
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net
import numpy as np
from timeit import default_timer as timer
import sys
import warnings
warnings.filterwarnings("ignore")

run_test = True


nb_training_classes = 334
thresh_top_1 = 0.90

model_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/dense_net/1117_base_setup"



def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    top_1 = AverageMeter()
    top_3 = AverageMeter()
    top_5 = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
    # for batch_idx, batch in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)

        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        prec1, prec3, prec5 = accuracy(output.data, target, topk=(1, 3, 5))
        top_1.update(prec1.item(), input.size(0))
        top_3.update(prec3.item(), input.size(0))
        top_5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                            'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                            'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                            'Loss %.4f' % (losses.val),
                            'Error %.4f' % (error.val),
                            'TOP-1 %.4f' % (top_1.val),
                            'TOP-3 %.4f' % (top_3.val),
                            'TOP-5 %.4f' % (top_5.val)])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, \
           top_1.avg, top_3.avg, top_5.avg


def valid_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    top_1 = AverageMeter()
    top_3 = AverageMeter()
    top_5 = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)

            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            prec1, prec3, prec5 = accuracy(output.data, target, topk=(1, 3, 5))
            top_1.update(prec1.item(), input.size(0))
            top_3.update(prec3.item(), input.size(0))
            top_5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f' % (losses.val),
                    'Error %.4f' % (error.val),
                    'TOP-1 %.4f' % (top_1.val),
                    'TOP-3 %.4f' % (top_3.val),
                    'TOP-5 %.4f' % (top_5.val),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg, top_1.avg, top_3.avg, top_5.avg


def train(model, train_loader, valid_loader, test_loader, save, n_epochs=300,
          batch_size=2, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
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
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        _, train_loss, train_error, train_acc_top1, \
        train_acc_top3, train_acc_top5 = train_epoch(model=model_wrapper,
                                                        loader=train_loader,
                                                        optimizer=optimizer,
                                                        epoch=epoch,
                                                        n_epochs=n_epochs)

        scheduler.step()

        print("validation")
        _, valid_loss, valid_error, valid_acc_top1, \
        valid_acc_top3, valid_acc_top5, = valid_epoch(model=model_wrapper,
                                                        loader=valid_loader if valid_loader else test_loader,
                                                        is_test=(not valid_loader))

        # Determine if model is the best
        if valid_loader:
            if valid_error < best_error:
                best_error = valid_error
                print('New best error: %.4f' % best_error)
                torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
        else:
            torch.save(model.state_dict(), os.path.join(save, 'model.dat'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d, '
                    '%0.6f, %0.6f, %0.6f, %0.6f, %0.6f, '
                    '%0.5f, %0.5f, %0.6f, %0.6f, %0.6f,\n' % (
                (epoch + 1),
                train_loss, train_error, train_acc_top1, train_acc_top3, train_acc_top5,
                valid_loss, valid_error, valid_acc_top1, valid_acc_top3, valid_acc_top5
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = valid_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)


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



def demo(train_data_dir, test_known_dir, test_unknown_dir, save_dir, depth=100, growth_rate=12, efficient=True, use_valid=True,
         n_epochs=100, batch_size=32, seed=None):
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

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    # Data transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Datasets
    train_set = datasets.ImageFolder(train_data_dir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize]))

    test_set_known = datasets.ImageFolder(test_known_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))

    test_set_unknown = datasets.ImageFolder(test_unknown_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))



    valid_size = int(len(train_set) / 5)
    indices = torch.randperm(len(train_set))

    train_indices = indices[:len(indices) - valid_size]
    valid_indices = indices[len(indices) - valid_size:]

    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=1,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                                               num_workers=1,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(train_set,
                                            batch_size=1,
                                            sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
                                            num_workers=1,
                                            pin_memory=True)

    # print(len(train_loader))
    # print(len(val_loader))
    # sys.exit()

    test_known_loader = torch.utils.data.DataLoader(test_set_known,
                                              batch_size=1, shuffle=False,
                                              num_workers=1, pin_memory=True)

    test_unknown_loader = torch.utils.data.DataLoader(test_set_unknown,
                                                    batch_size=1, shuffle=False,
                                                    num_workers=1, pin_memory=True)

    # Models
    model = efficient_dense_net.DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate * 2,
        num_classes=335,
        small_inputs=True,
        efficient=efficient)

    if run_test:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.dat')))

        print("*" * 50)
        print("Testing the known samples...")
        test_with_novelty(test_loader=test_known_loader,
                          model=model,
                          test_unknown=False)

        print("*" * 50)
        print("testing the unknown samples...")
        test_with_novelty(test_loader=test_unknown_loader,
                          model=model,
                          test_unknown=True)
        print("*" * 50)

        return


    else:
        # Print number of parameters
        num_params = sum(p.numel() for p in model.parameters())
        print("Total parameters: ", num_params)

        # Make save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.isdir(save_dir):
            raise Exception('%s is not a dir' % save_dir)

        # Train the model
        train(model=model, train_loader=train_loader, valid_loader=val_loader, test_loader=test_loader, save=save_dir,
              n_epochs=n_epochs, batch_size=batch_size, seed=seed)
        print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.
Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>
Try out the naive DenseNet implementation:
python demo.py --efficient False --data <path_to_data_dir> --save <path_to_save_dir>
Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
if __name__ == '__main__':
    demo(train_data_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/debug/train_valid/known_known",
         test_known_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/debug/test/known_known",
         test_unknown_dir = "/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_3_partition/debug/test/unknown_unknown",
         save_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/dense_net/1117_base_setup")