# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import re
import argparse
import os
import shutil
import time
import math
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torchvision.datasets

from mean_teacher import architectures, datasets, data, losses, ramps, cli
from mean_teacher.run_context import RunContext
from mean_teacher.data import NO_LABEL
from mean_teacher.utils import *
from model import ResNet18withSobel
from dataloader import *



args = None
best_acc = 0
global_step = 0

    



def main(context):
    global best_acc
    dataset_config = datasets.__dict__[args.dataset]()
    num_classes = dataset_config.pop('num_classes')
    label_loader,unlabel_loader, eval_loader = create_data_loaders(**dataset_config, args=args)
    args.lr =  0.03
    
    print(args.lr)
    print(args.momentum)
    def create_model(ema=False):
        net = ResNet18withSobel()
        net = torch.nn.DataParallel(net)
        checkpoint = torch.load('./checkpoint/ckpt_cifar.t7')
        net.load_state_dict(checkpoint['net'])
        fc = nn.Linear(128, 10)
        model = nn.Sequential(net, fc)
        model = model.to('cuda')
        
        if ema:
            for param in model.parameters():
                param.detach_()                
                
        return model

    model = create_model()
    ema_model = create_model(ema = True)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    
    for epoch in range(args.start_epoch, 180):
        train(label_loader, unlabel_loader, model, ema_model, optimizer, epoch)
        acc = validate(eval_loader, model)
#         acc2 = validate(eval_loader, ema_model)
        
        
        if acc > best_acc:
            print("state saving...")
            state = {
                    'model': model.state_dict(),
                    'epoch': epoch
                    }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
#             torch.save(state, './checkpoint/ckpt_svhn_no_init.t7')
            best_acc = acc

        print('Best Acc %.2f %%' % (100 * best_acc))
    

def parse_dict_args(**kwargs):
    global args

    def to_cmdline_kwarg(key, value):
        if len(key) == 1:
            key = "-{}".format(key)
        else:
            key = "--{}".format(re.sub(r"_", "-", key))
        value = str(value)
        return key, value

    kwargs_pairs = (to_cmdline_kwarg(key, value)
                    for key, value in kwargs.items())
    cmdline_args = list(sum(kwargs_pairs, ()))
    args = parser.parse_args(cmdline_args)


def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir,
                        args):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.RandomResizedCrop(size=32, scale=(0.2,1.)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
#         transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
        transforms.Normalize(mean, std),

    ])

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
#         transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1201, 0.1231, 0.1052)),
        transforms.Normalize(mean, std),

    ])
        # Train and Test is same dataset but data augmentation is different
    labeled = CIFAR_SSL('./data', split='label', download=False, transform=transform_train, boundary=0)
    unlabeled = CIFAR_SSL('./data', split='unlabel', download=False, transform=transform_train, boundary=0)
    testset = CIFAR_SSL('./data', split='test', download=False, transform=transform_test, boundary=0)
    label_loader = torch.utils.data.DataLoader(labeled, batch_size=100, shuffle=True, num_workers=4, drop_last=False)
    unlabel_loader = torch.utils.data.DataLoader(unlabeled, batch_size=256, shuffle=True, num_workers=4, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4, drop_last=False)
        
    return label_loader, unlabel_loader, testloader




def train(label_loader, unlabel_loader, model,ema_model, optimizer, epoch):
    adjust_learning_rate(optimizer, epoch)
    global global_step
    class_criterion = nn.CrossEntropyLoss(size_average=False).cuda()    
    consistency_criterion = losses.softmax_mse_loss
    criterion_l1 = nn.L1Loss(size_average=False).cuda()

    # switch to train mode
    model.train()
    ema_model.train()
    
    label_iter = iter(label_loader)     
    unlabel_iter = iter(unlabel_loader)     
    len_iter = len(unlabel_iter)
    for i, (input, target,_) in enumerate(label_loader):
        input_var = torch.autograd.Variable(input).to('cuda')
        target_var = torch.autograd.Variable(target).to('cuda')
        outputs = model(input_var)
        batch_size = outputs.shape[0]
        loss = class_criterion(outputs, target_var) / float(batch_size)
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
#         update_ema_variables(model, ema_model, args.ema_decay, global_step)


        if i % 80 == 0:
             print('[Epoch : %d, %5d] train loss: %.4f' % (epoch + 1, i + 1, float(loss.item())))
        
        

def validate(eval_loader, model):
    # switch to evaluate mode
    model.eval()
    correct = 0
    total = 0
    
    for i, (input, target,_) in enumerate(eval_loader):
        input_var = torch.autograd.Variable(input).to('cuda')
        target_var = torch.autograd.Variable(target).to('cuda')
        outputs = model(input_var)
        _, predicted = torch.max(outputs.data, 1)
        total += target_var.size(0)
        correct += (predicted == target_var).sum().item()
    print('Accuracy %.2f %%' % (100 * correct / total))
    
    return correct / total



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 40:
        lr = args.lr * (0.1 ** ((epoch - 40) // 40))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 100.0 * ramps.sigmoid_rampup(epoch, 5)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def cal_reg_l1(model, criterion_l1):
    reg_loss = 0
    np = 0
    for param in model.parameters():
        reg_loss += criterion_l1(param, torch.zeros_like(param))
        np += param.nelement()
    reg_loss = reg_loss / np
    return reg_loss        


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = cli.parse_commandline_args()
    main(RunContext(__file__, 0))
