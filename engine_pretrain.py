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

import umap
import umap.plot


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
                    queue=None,
                    global_rank=0):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    
    if args.bt_loss_coef_decay == 'exp':
        bt_coef = exp_scheduler(epoch, args.epochs) * args.bt_loss_coef
        print('exp decay bt loss coef')
    elif args.bt_loss_coef_decay == 'cosine':
        bt_coef = cosine_scheduler(epoch, args.epochs) * args.bt_loss_coef
        print('cosine decay bt loss coef')
    else:
        bt_coef = args.bt_loss_coef

    if queue is None and args.bt_nn:
        queue = torch.rand((args.bt_nn_queue_size, model.patch_embed.proj.out_channels)).to(args.device)
        new_queue = torch.zeros((args.bt_nn_queue_size, model.patch_embed.proj.out_channels)).to(args.device)
    elif args.bt_nn:
        new_queue = torch.zeros((args.bt_nn_queue_size, model.patch_embed.proj.out_channels)).to(args.device)
    else:
        new_queue = None

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

        return_extras = {}

    projector_dim = int(args.projector.split('-')[-1])
    accum_matrix = torch.tensor(np.zeros((projector_dim, projector_dim))).to(device)
    accum_on = torch.tensor(0).to(device)
    accum_off = torch.tensor(0).to(device)
    for data_iter_step, ((x1, x2), _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        x1 = x1.to(device, non_blocking=True)
        x2 = x2.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            mae_loss1, mae_loss2, bt_loss,c, bt_mixup_loss, on, off, pred1, pred2, mask1, mask2, latent1, latent2 = model(x1, x2, mask_ratio=args.mask_ratio, bt_coef=bt_coef, bt_mode=args.bt_mode, bt_mixup = args.bt_mixup, bt_mixup_loss_scale = args.bt_mixup_loss_scale, bt_global_pooling=args.bt_global_pooling, prev_iteractions=queue)
            loss = mae_loss1 + mae_loss2 + bt_loss + bt_mixup_loss
            #print('mae_loss1: {}, mae_loss2: {}, bt_loss: {}, on:{}, off: {}'.format(mae_loss1.item(), mae_loss2.item(), bt_loss.item(), on.item(), off.item()))

        #edited latent space
        latent2_e = None
        if args.bt_nn and args.bt_global_pooling:
            latent2_e = latent2[:, 1:, :].mean(dim=1)
        elif args.bt_nn and not args.bt_global_pooling:
            latent2_e = latent2[:, 0, :]
        

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
        metric_logger.update(bt_mixup_loss=bt_mixup_loss.item())
        metric_logger.update(bt_coef=bt_coef)

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
        
        #accum_matrix = np.sum([accum_matrix, c.detach().cpu().numpy()], axis=0)
        accum_matrix = torch.add(accum_matrix, c)
        accum_on = torch.add(accum_on, on)
        accum_off = torch.add(accum_off, off)

        
        if(data_iter_step >= (len(data_loader) - (args.bt_nn_queue_size/args.batch_size))) and(not latent2_e is None):
            new_queue = torch.cat((new_queue[args.batch_size:], latent2_e))
            
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    
    accum_matrix = accum_matrix.detach().cpu().numpy() / (data_iter_step + 1)
    accum_on = accum_on.detach().cpu().numpy() / (data_iter_step + 1)
    accum_off = accum_off.detach().cpu().numpy() / (data_iter_step + 1)

    metric_logger.update(bt_on_diag=accum_on)
    metric_logger.update(bt_off_diag=accum_off)

    return_extras['accum_matrix'] = {'value':accum_matrix, 'type':'txt'}

    
    if global_rank == 0 and args.knn_eval:
        train_features, train_labels = extract_features(model, data_loader_eval_train, device)
        val_features, val_labels = extract_features(model, data_loader_eval_val, device)
        
        knn = train_knn_classifier(train_features, train_labels)
        f1, accuracy = evaluate_knn_classifier(knn, val_features, val_labels)
        #print('KNN evaluation: f1: {}, accuracy: {}'.format(f1, accuracy))
        metric_logger.update(knn_f1_val=f1)
        metric_logger.update(knn_accuracy_val=accuracy)
        

        #if True:
        if (epoch+1) % 100 == 0:
            #train_mapper = umap.UMAP().fit(train_features)
            val_mapper = umap.UMAP().fit(val_features)

            #train_image = umap.plot.points(train_mapper, labels=train_labels)
            val_image = umap.plot.points(val_mapper, labels=val_labels)

            #return_extras['umap_train_image'] = {'value':train_image, 'type':'axis'}
            return_extras['umap_val_image'] = {'value':val_image, 'type':'axis'}
        

    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, return_extras, new_queue


def extract_features(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
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

    return features, labels


#method that train a k-nn classifier based on the features extracted from the model
def train_knn_classifier(features, labels, k: int = 5):
    # Train k-NN classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features, labels)

    return knn

def evaluate_knn_classifier(knn, features, true_labels):
    # Predict using k-NN classifier
    pred_labels = knn.predict(features)
    f1 = f1_score(true_labels, pred_labels, average='macro')
    accuracy = accuracy_score(true_labels, pred_labels)

    return f1, accuracy

