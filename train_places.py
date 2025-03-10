import argparse
import os
import sys
import gc
import shutil
import time
import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from MODELS.model_resnet import *
from plot_functions import *
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Places365 Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet',
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--whitened_layers', default='8')
parser.add_argument('--act_mode', default='pool_max')
parser.add_argument('--depth', default=18, type=int, metavar='D',
                    help='model depth')
parser.add_argument('--ngpu', default=4, type=int, metavar='G',
                    help='number of gpus to use')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--concepts', type=str, required=True)
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument("--seed", type=int, default=1234, metavar='BS',
                    help='input batch size for training (default: 64)')
parser.add_argument("--prefix", type=str, required=True,
                    metavar='PFX', help='prefix for logging & checkpoint saving')
parser.add_argument('--evaluate', dest='evaluate',
                    action='store_true', help='evaluation only')
best_prec1 = 0

os.chdir(sys.path[0])

if not os.path.exists('./checkpoints'):
    os.mkdir('./checkpoints')


def main():
    global args, best_prec1
    args = parser.parse_args()

    print("args", args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    args.prefix += '_' + '_'.join(args.whitened_layers.split(','))

    # create model
    if args.arch == "resnet_cw":
        if args.depth == 50:
            model = ResidualNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(
                ',')], arch='resnet50', layers=[3, 4, 6, 3], model_file='./checkpoints/resnet50_places365.pth.tar')
        elif args.depth == 18:
            model = ResidualNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(
                ',')], arch='resnet18', layers=[2, 2, 2, 2], model_file='./checkpoints/resnet18_places365.pth.tar')
    elif args.arch == "resnet_original" or args.arch == "resnet_baseline":
        if args.depth == 50:
            model = ResidualNetBN(365, args, arch='resnet50', layers=[
                                  3, 4, 6, 3], model_file='./checkpoints/resnet50_places365.pth.tar')
        if args.depth == 18:
            model = ResidualNetBN(365, args, arch='resnet18', layers=[
                                  2, 2, 2, 2], model_file='./checkpoints/resnet18_places365.pth.tar')
    elif args.arch == "densenet_cw":
        model = DenseNetTransfer(365, args, [int(x) for x in args.whitened_layers.split(
            ',')], arch='densenet161', model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == 'densenet_original':
        model = DenseNetBN(365, args, arch='densenet161',
                           model_file='./checkpoints/densenet161_places365.pth.tar')
    elif args.arch == "vgg16_cw":
        model = VGGBNTransfer(365, args, [int(x) for x in args.whitened_layers.split(
            ',')], arch='vgg16_bn', model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')
    elif args.arch == "vgg16_bn_original":
        # 'vgg16_bn_places365.pt')
        model = VGGBN(365, args, arch='vgg16_bn',
                      model_file='./checkpoints/vgg16_bn_places365_12_model_best.pth.tar')

    # print(args.start_epoch, args.best_prec1)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    # param_list = get_param_list_bn(model)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
    model = model.cuda()
    print("model")
    print(model)

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    testdir = os.path.join(args.data, 'test')
    conceptdir_train = os.path.join(args.data, 'concept_train')
    conceptdir_test = os.path.join(args.data, 'concept_test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    concept_loaders = [
        torch.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(conceptdir_train, concept),
                transforms.Compose([
                    transforms.RandomSizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False) for concept in args.concepts.split(',')]

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize])),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False)

    # val_loader_2 = torch.utils.data.DataLoader(
    #     datasets.ImageFolder('/usr/xtmp/zhichen/ConceptWhitening_git/ConceptWhitening/plot/airplane_bed_bench_boat_book_horse_person/resnet_cw18/1_rot_cw_top5', transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)

    # test_loader = torch.utils.data.DataLoader(
    #     datasets.ImageFolder(testdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.workers, pin_memory=False)

    # test_loader_with_path = torch.utils.data.DataLoader(
    #     ImageFolderWithPaths(testdir, transforms.Compose([
    #         transforms.Scale(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         normalize,
    #     ])),
    #     batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=False)

    if args.evaluate == False:
        print("Start training")
        best_prec1 = 0
        for epoch in range(args.start_epoch, args.start_epoch + 2):
            adjust_learning_rate(optimizer, epoch)

            # train for one epoch
            if args.arch == "resnet_cw":
                train(train_loader, concept_loaders,
                      model, criterion, optimizer, epoch)
            elif args.arch == "resnet_baseline":
                train_baseline(train_loader, concept_loaders,
                               model, criterion, optimizer, epoch)
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, args.prefix)
        print(best_prec1)
        # validate(test_loader, model, criterion, epoch)
    else:
        # model = load_resnet_model(args, arch = 'resnet_baseline', depth=args.depth, whitened_layer=args.whitened_layers)
        # print('resnet_orginal')
        # for loader in concept_loaders:
        #     get_representation_distance_to_center(args, loader, '8', arch='resnet_original')
        # print('resnet_cw')
        # for loader in concept_loaders:
        #     get_representation_distance_to_center(args, loader, '8', arch='resnet_cw')
        # intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir_test, '8', plot_cpt = args.concepts.split(','), arch='resnet_cw')
        # intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir_test, '8', plot_cpt = args.concepts.split(','), arch='resnet_baseline')
        # intra_concept_dot_product_vs_inter_concept_dot_product(args, conceptdir_test, '8', plot_cpt = args.concepts.split(','), arch='resnet_original')

        # print("Start testing")
        # # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='8')
        # # validate(test_loader, model, criterion, args.start_epoch)
        # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='7')
        # validate(test_loader, model, criterion, args.start_epoch)
        # # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='6')
        # # validate(test_loader, model, criterion, args.start_epoch)
        # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='5')
        # validate(test_loader, model, criterion, args.start_epoch)
        # # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='4')
        # # validate(test_loader, model, criterion, args.start_epoch)
        # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='3')
        # validate(test_loader, model, criterion, args.start_epoch)
        # # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='2')
        # # validate(test_loader, model, criterion, args.start_epoch)
        # model = load_resnet_model(args, arch = args.arch, depth=args.depth, whitened_layer='1')
        # validate(test_loader, model, criterion, args.start_epoch)
        # print("Start Plotting")
        # plot_figures(args, model, test_loader_with_path, train_loader, concept_loaders, conceptdir_test)
        # saliency_map_concept_cover(args, val_loader_2, '1', arch='resnet_cw', dataset='places365', num_concepts=7)

        print("Start Plotting")
        times = ['20210903_073210',
                 '20210904_115816',
                 '20210904_205429',
                 '20210905_131108']
        for time in times:
            print("Loading model {}_{}".format('_'.join(args.concepts.split(',')),time))

            model = load_resnet_model(args, time, arch=args.arch, depth=args.depth, whitened_layer='7')

            plot_figures(args, model, val_loader, train_loader, concept_loaders, conceptdir_test, time)
        # saliency_map_concept_cover(
        #     args, val_loader, '7', arch='resnet_cw', dataset='places365',
        #     num_concepts=9)
        pass


def train(train_loader, concept_loaders, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # if (i + 1) % 800 == 0:
        #     break
        if (i + 1) % 30 == 0:
            model.eval()
            with torch.no_grad():
                # update the gradient matrix G
                for concept_index, concept_loader in enumerate(concept_loaders):
                    model.module.change_mode(concept_index)
                    for j, (X, _) in enumerate(concept_loader):
                        X_var = torch.autograd.Variable(X).cuda()
                        model(X_var)
                        break
                model.module.update_rotation_matrix()
                # change to ordinary mode
                model.module.change_mode(-1)
            model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    return top1.avg


'''
This function train a baseline with auxiliary concept loss jointly
train with main objective
'''


def train_baseline(train_loader, concept_loaders, model, criterion, optimizer, epoch, activation_mode='pool_max'):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_aux = AverageMeter()
    top1_cpt = AverageMeter()

    n_cpt = len(concept_loaders)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # switch to train mode
    model.train()

    end = time.time()

    inter_feature = []

    def hookf(module, input, output):
        inter_feature.append(output[:, :n_cpt, :, :])
    for i, (input, target) in enumerate(train_loader):
        if (i + 1) % 20 == 0:

            # model.eval()

            layer = int(args.whitened_layers)
            layers = model.module.layers
            if layer <= layers[0]:
                hook = model.module.model.layer1[layer -
                                                 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1]:
                hook = model.module.model.layer2[layer -
                                                 layers[0] - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2]:
                hook = model.module.model.layer3[layer - layers[0] -
                                                 layers[1] - 1].bn1.register_forward_hook(hookf)
            elif layer <= layers[0] + layers[1] + layers[2] + layers[3]:
                hook = model.module.model.layer4[layer - layers[0] -
                                                 layers[1] - layers[2] - 1].bn1.register_forward_hook(hookf)

            y = []
            inter_feature = []
            for concept_index, concept_loader in enumerate(concept_loaders):
                for j, (X, _) in enumerate(concept_loader):
                    y += [concept_index] * X.size(0)
                    X_var = torch.autograd.Variable(X).cuda()
                    model(X_var)
                    break

            inter_feature = torch.cat(inter_feature, 0)
            y_var = torch.Tensor(y).long().cuda()
            f_size = inter_feature.size()
            if activation_mode == 'mean':
                y_pred = F.avg_pool2d(inter_feature, f_size[2:]).squeeze()
            elif activation_mode == 'max':
                y_pred = F.max_pool2d(inter_feature, f_size[2:]).squeeze()
            elif activation_mode == 'pos_mean':
                y_pred = F.avg_pool2d(
                    F.relu(inter_feature), f_size[2:]).squeeze()
            elif activation_mode == 'pool_max':
                kernel_size = 3
                y_pred = F.max_pool2d(inter_feature, kernel_size)
                y_pred = F.avg_pool2d(y_pred, y_pred.size()[2:]).squeeze()

            loss_cpt = 10 * criterion(y_pred, y_var)
            # measure accuracy and record loss
            [prec1_cpt] = accuracy(y_pred.data, y_var, topk=(1,))
            loss_aux.update(loss_cpt.data, f_size[0])
            top1_cpt.update(prec1_cpt[0], f_size[0])

            optimizer.zero_grad()
            loss_cpt.backward()
            optimizer.step()

            hook.remove()
            # model.train()
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Loss_aux {loss_a.val:.4f} ({loss_a.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                  'Prec_cpt@1 {top1_cpt.val:.3f} ({top1_cpt.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, loss_a=loss_aux, top1=top1, top5=top5, top1_cpt=top1_cpt))


def plot_figures(
        args, model, test_loader_with_path, train_loader, concept_loaders,
        conceptdir, time):
    concept_name = args.concepts.split(',')

    path_plots = './plots/' + '_'.join(concept_name) + "_{}/".format(time)
    if not os.path.exists(path_plots):
        os.mkdir(path_plots)

    print("Plot top50 activated images")
    plot_concept_top50(
        args, test_loader_with_path, model, '7', path_plots,
        activation_mode=args.act_mode)

    pairs = list(itertools.combinations(concepts_test, 2))
    
    print("Plot 2d slice of representation within the same topic")
    for pair in pairs:
        print("Concepts: {} :: {}".format(pair[0], pair[1]))
        plot_concept_representation(
            args, test_loader_with_path, model, '7', path_plots,
            plot_cpt=[pair[0], pair[1]], activation_mode=args.act_mode)

    print("Plot correlation")
    plot_correlation(args, test_loader_with_path, model, 7, path_plots)

    print("Plot trajectory")
    for pair in pairs:
        print("Concepts: {} :: {}".format(pair[0], pair[1]))
        plot_trajectory(
            args, time, test_loader_with_path, '7', path_plots,
            plot_cpt=[pair[0], pair[1]])

    print("Plot AUC-concept_purity")
    aucs_cw = plot_auc_cw(
        args, time, conceptdir, '7', path_plots, 
        plot_cpt=concept_name, activation_mode=args.act_mode)

    print("End plotting")


def save_checkpoint(state, is_best, prefix, checkpoint_folder='./checkpoints'):
    concept_name = '_'.join(args.concepts.split(',')) + "_" + time.strftime("%Y%m%d_%H%M%S")
    if not os.path.exists(os.path.join(checkpoint_folder, concept_name)):
        os.mkdir(os.path.join(checkpoint_folder, concept_name))
    filename = os.path.join(checkpoint_folder, concept_name,
                            '%s_checkpoint.pth.tar' % prefix)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(
            checkpoint_folder, concept_name, '%s_model_best.pth.tar' % prefix))


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs

    """
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
