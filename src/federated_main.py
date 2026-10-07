#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
#import pickle
import csv  
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    device = 'cuda' if args.gpu else 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    # train_loss, train_accuracy = [], []
    train_loss, test_accuracy = [], []
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    # print_every = 2
    # val_loss_pre, counter = 0, 0

    for epoch in tqdm(range(args.epochs)):
        # init local weights and loss
        local_weights, local_losses = [], []
        # print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        # sample a fraction of users (with args frac)
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)

        # for each sampled user
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        # loss_avg = sum(local_losses) / len(local_losses)
        # train_loss.append(loss_avg)

        # # Calculate avg training accuracy over all users at every epoch
        # list_acc, list_loss = [], []
        # global_model.eval()
        # for c in range(args.num_users):
        #     local_model = LocalUpdate(args=args, dataset=train_dataset,
        #                               idxs=user_groups[idx], logger=logger)
        #     acc, loss = local_model.inference(model=global_model)
        #     list_acc.append(acc)
        #     list_loss.append(loss)
        # train_accuracy.append(sum(list_acc)/len(list_acc))

        # # print global training loss after every 'i' rounds
        # if (epoch+1) % print_every == 0:
        #     print(f' \nAvg Training Stats after {epoch+1} global rounds:')
        #     print(f'Training Loss : {np.mean(np.array(train_loss))}')
        #     print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
        train_loss.append(sum(local_losses) / len(local_losses))

        acc, _ = test_inference(args, global_model, test_dataset)
        test_accuracy.append(acc)

        if acc >= args.target:
            break

    # Test inference after completion of training
    print(f'\n Rounds completed: {len(test_accuracy)}')
    print('|---- Final test accuracy: {:.2f}%'.format(100 * test_accuracy[-1]))
    print('|---- Best test accuracy: {:.2f}%'.format(100 * max(test_accuracy)))

    os.makedirs('save/results', exist_ok=True)
    file_name = 'save/results/{}_iid{}_C{}_B{}_E{}_lr{}.csv'.format(
        args.model, args.iid, args.frac, args.local_bs, args.local_ep, args.lr)

    with open(file_name, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['round', 'test_accuracy', 'train_loss'])
        for r, (acc, loss) in enumerate(zip(test_accuracy, train_loss), start=1):
            writer.writerow([r, acc, loss])

    print(f' Saved: {file_name}')
    print(' Total run time: {0:0.4f}'.format(time.time() - start_time))
