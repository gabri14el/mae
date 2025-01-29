# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import f1_score

import util.misc as misc
import util.lr_sched as lr_sched
from util.bt_sched import cosine_scheduler, exp_scheduler


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_dual(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    data_loader_eval_train=None,
                    data_loader_eval_val=None,
                    global_rank=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    
    if args.bt_loss_coef_decay == 'exp':
        bt_coef = exp_scheduler(epoch, args.epochs) * args.bt_loss_coef
    elif args.bt_loss_coef_decay == 'cosine':
        bt_coef = cosine_scheduler(epoch, args.epochs) * args.bt_loss_coef
    else:
        bt_coef = args.bt_loss_coef
    

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, ((x1, x2), _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            mae_loss1, mae_loss2, bt_loss, pred1, pred2, mask1, mask2, latent1, latent2 = model(x1, x2, mask_ratio=args.mask_ratio, bt_coef=bt_coef)
            loss = mae_loss1 + mae_loss2 + bt_loss
            #print('mae_loss1: {}, mae_loss2: {}, bt_loss: {}'.format(mae_loss1.item(), mae_loss2.item(), bt_loss.item()))

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(mae_loss_1=mae_loss1.item())
        metric_logger.update(mae_loss_2=mae_loss2.item())
        metric_logger.update(bt_loss=bt_loss.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    

    
    if global_rank == 0 and args.knn_eval:
        knn = train_knn_classifier(model, data_loader_eval_train, device)
        f1, accuracy = evaluate_knn_classifier(knn, model, data_loader_eval_val, device)
        #print('KNN evaluation: f1: {}, accuracy: {}'.format(f1, accuracy))
        metric_logger.update(knn_f1_val=f1)
        metric_logger.update(knn_accuracy_val=accuracy)

    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#method that train a k-nn classifier based on the features extracted from the model
def train_knn_classifier(model, data_loader: Iterable, device: torch.device, k: int = 5):
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for (samples, targets) in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            #Extract features from the model
            feature = model.forward_representations(samples)
            features.append(feature.cpu().numpy())
            labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)

    return knn

def evaluate_knn_classifier(knn, model, data_loader: Iterable, device: torch.device):
    model.eval()
    features = []
    true_labels = []

    with torch.no_grad():
        for (samples, targets) in data_loader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Extract features from the model
            feature = model.forward_representations(samples)
            features.append(feature.cpu().numpy())
            true_labels.append(targets.cpu().numpy())

    features = np.concatenate(features, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)

    # Predict using k-NN classifier
    pred_labels = knn.predict(features)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)

    return f1, accuracy