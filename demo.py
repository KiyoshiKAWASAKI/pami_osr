import fire
import os
import time
import torch
from torchvision import datasets, transforms
from models import efficient_dense_net



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
                    '%0.5f, %0.5f,\n' % (
                (epoch + 1),
                train_loss, train_error, train_acc_top1, train_acc_top3, train_acc_top5,
                valid_loss, valid_error,
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



def demo(train_data_dir, test_data_dir, save_dir, depth=100, growth_rate=12, efficient=True, use_valid=True,
         n_epochs=100, batch_size=8, seed=None):
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

    test_set = datasets.ImageFolder(test_data_dir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize]))


    if use_valid:
        valid_size = int(len(train_set) / 5)
        indices = torch.randperm(len(train_set))

        train_indices = indices[:len(indices) - valid_size]
        valid_indices = indices[len(indices) - valid_size:]

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size,
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
                                                   num_workers=1,
                                                   pin_memory=True)

        val_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
                                                num_workers=1,
                                                pin_memory=True)

        test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=1, pin_memory=True)


    else:
        valid_loader = None

    # Models
    model = efficient_dense_net.DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate * 2,
        num_classes=7,
        small_inputs=True,
        efficient=efficient,
    )
    print(model)

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
    demo(train_data_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_baseline_0722/small_data_for_debug/train",
         test_data_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/data/object_recognition/image_net/derivatives/dataset_v1_baseline_0722/small_data_for_debug/val",
         save_dir="/afs/crc.nd.edu/user/j/jhuang24/scratch_22/open_set/models/sail-on/dense_net/1117_base_setup")