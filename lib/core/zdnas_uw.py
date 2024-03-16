# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Written by Hao Du and Houwen Peng
# email: haodu8-c@my.cityu.edu.hk and houwen.peng@microsoft.com

from pathlib import Path
import numpy as np
import time, os
import torchvision
import torch.nn.functional as F
import itertools
import random as rd
import copy
import operator
from tqdm import tqdm
from torch.cuda import amp
from scipy.special import softmax
from lib.utils.util import *
from lib.utils.general import compute_loss, test, plot_images, is_parallel, build_foreground_mask, compute_sensitive_loss
from lib.utils.kd_utils import compute_loss_KD
from lib.utils.synflow import sum_arr_tensor
from lib.zero_proxy import snip, synflow, naswot, grasp
from datetime import datetime


PROXY_DICT = {
    # 'snip'  :  snip.calculate_snip,
    # 'synflow': synflow.calculate_synflow,
    'naswot' : naswot.calculate_zero_cost_map2,
    # 'grasp': grasp.calculate_grasp
}

def focal_loss_dtp(acc, k, alpha=0.95, gamma=2.0):
    # acc = 0-1
    k = alpha * acc + (1-alpha) * k
    return k, -torch.pow(1-k, gamma) * torch.log(k)

#######################################
# Search Zero-Cost Aging Evolution
#######################################
def train_epoch_zdnas_uw(epoch, model, zc_func, theta_optimizer, cfg, device, task_flops, 
                     est=None, table=None, logger=None, local_rank=0, prefix='', logdir='./', stage=9):
    batch_size = cfg.DATASET.BATCH_SIZE
    is_ddp = is_parallel(model)
    nn_model = model.module if is_ddp else model
    
    naive_model  = model.module if is_ddp else model
    search_space = naive_model.search_space
    temperature = nn_model.temperature
    
    # Average Meter
    avg_params = AverageMeter()
    avg_flops  = AverageMeter()
    avg_floss  = AverageMeter()
    avg_zcloss = AverageMeter()
    avg_depth = AverageMeter()
    avg_latency = AverageMeter()
    total      = AverageMeter()
    # print(list(model.parameters()))
    
    # k_flops = 0.
    # k_latency = 0.
    # k_zc = 0.
    print(f"uw: {model.k_depth}, {model.k_latency}, {model.k_zc}")

    depth_max = 111
    depth_min = 5
    latency_max = 565
    latency_min = 52
    alpha = 10 #0.01 #0.01 # 0.03        # for flops_loss
    # beta  = 0.01         # for params_loss
    # gamma = 1 #0.03 # 0.01        # for zero cost loss
    # omega = 0.01        # for depth loss
    # theta = 1 #0.01 # 0.005 # for latency loss
    num_iter = 2880
    if local_rank in [-1, 0]:
        # logger.info(('%10s' * 9) % ('Epoch', 'gpu_num', 'Param', 'FLOPS', 'Latency', 'f_loss', 'zc_loss', 'total', 'temp'))
        logger.info(('%10s' * 10) % ('Epoch', 'gpu_num', 'Param', 'FLOPS', 'Depth', 'Latency', 'f_loss', 'zc_loss', 'total', 'temp'))
        pbar = tqdm(range(num_iter), total=num_iter, bar_format='{l_bar}{bar:5}{r_bar}')  # progress bar
    
    f=open(os.path.join(logdir, 'train.txt'), 'a')
    for iter_idx in pbar:
        if iter_idx % 50 == 0: 
            arch_prob = model.module.softmax_sampling(temperature, detach=True) if is_ddp else model.softmax_sampling(temperature, detach=True)
            zc_map = zc_func(arch_prob)
            f.write(f'[{epoch}-{iter_idx:04d}] {str(arch_prob)}\n')
            f.write(f'[{epoch}-{iter_idx:04d}] {str(zc_map)}\n')
            
        ##########################################################
        # Calculate Basic Information (FLOPS, Params, ZC_Score)
        ##########################################################
        gumbel_prob = model.module.gumbel_sampling(temperature) if is_ddp else model.gumbel_sampling(temperature)
        architecture_info = {
            'arch_type': 'continuous',
            'arch': gumbel_prob
        }
         
        zc_score, largest_zc = nn_model.calculate_zc2(architecture_info, zc_map)
        output_flops  = nn_model.calculate_flops_new(architecture_info, est.flops_dict) / 1e3
        output_params = nn_model.calculate_params_new(architecture_info, est.params_dict)
        output_depth = nn_model.calculate_layers_new(architecture_info)
        output_latency = nn_model.calculate_table_latency(architecture_info, table)
        
        #########################################
        # Calculate Loss
        #########################################
        squared_error_flops = (output_flops - task_flops) ** 2
        # squared_error_params = (output_params - task_params) ** 2
        
        # depth_acc = (depth_max - output_depth) / (depth_max - depth_min)
        # flops_acc = 1 - (abs(task_flops - output_flops) / task_flops)
        latency_acc = (latency_max - output_latency) / (latency_max - latency_min)
        zc_acc = zc_score / (largest_zc*stage)
        print(f"latency_acc: {latency_acc}, zc_acc: {zc_acc}")
        # print(f"depth_acc: {depth_acc}, latency_acc: {latency_acc}, zc_acc: {zc_acc}")

        # model.k_depth, depth_fl = focal_loss_dtp(depth_acc, model.k_depth)
        # model.k_flops, flops_fl = focal_loss_dtp(flops_acc, model.k_flops)
        model.k_latency, latency_fl = focal_loss_dtp(latency_acc, model.k_latency, alpha=0.5, gamma=1)
        model.k_zc, zc_fl = focal_loss_dtp(zc_acc, model.k_zc, alpha=0.5, gamma=1)
        print(f"latency_fl: {latency_fl}, zc_fl: {zc_fl}")
        # print(f"depth_fl: {depth_fl}, latency_fl: {latency_fl}, zc_fl: {zc_fl}")

        # depth_loss = output_depth * depth_fl.item()
        # flops_loss = squared_error_flops * flops_fl.item()
        latency_loss = output_latency * latency_fl.item()
        zc_loss = zc_score * zc_fl.item()

        # flops_loss = squared_error_flops / (2*model.loss_scale[0].exp()) + model.loss_scale[0] #regration
        # latency_loss = output_latency / (2*model.loss_scale[1].exp()) + model.loss_scale[1]
        # zc_loss = zc_score / (2*model.loss_scale[1].exp()) + model.loss_scale[1] #classification

        flops_loss = squared_error_flops * alpha
        # # params_loss = squared_error_params * beta
        # # layers_loss = output_layers * omega
        # latency_loss = output_latency * theta
        # zc_loss     = zc_score * gamma
        
        loss = zc_loss + flops_loss + latency_loss
        # loss = zc_loss + flops_loss + depth_loss + latency_loss

        #########################################
        # Calculate Loss
        #########################################
        theta_optimizer.zero_grad()
        loss.backward()
        theta_optimizer.step()
        
        # Update Average Meter
        avg_params.update(output_params.item(), 1)
        avg_flops.update(output_flops.item(), 1)
        avg_depth.update(output_depth.item(), 1)
        avg_latency.update(output_latency.item(), 1)
        avg_floss.update(squared_error_flops.item(), 1)
        avg_zcloss.update(zc_score.item(), 1)
        total.update(loss.item(), 1)
        
        # Print
        if local_rank in [-1, 0]:
            ni = iter_idx
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            # s = ('%10s' * 2 + '%10.4g' * 7) % (
            #     '%3d/%3d' % (epoch,cfg.EPOCHS), mem, avg_params.avg, avg_flops.avg, avg_latency.avg, avg_floss.avg, \
            #         avg_zcloss.avg, total.avg, temperature)
            s = ('%10s' * 2 + '%10.4g' * 8) % (
                '%3d/%3d' % (epoch,cfg.EPOCHS), mem, avg_params.avg, avg_flops.avg, avg_depth.avg, avg_latency.avg, avg_floss.avg, \
                    avg_zcloss.avg, total.avg, temperature)
            # uw = ("uncertainty weights: flop = %10.4g, zc = %10.4g")% (1/(2*np.exp(model.loss_scale[0].item())), 1/(2*np.exp(model.loss_scale[1].item())))#, 1/(2*np.exp(model.loss_scale[2].item())))  , latency = %10.4g
            uw = ("DTP: latency = %10.4g, zc = %10.4g") % (latency_fl.item(), zc_fl.item())
            # uw = ("DTP: depth = %10.4g, latency = %10.4g, zc = %10.4g") % (depth_fl.item(), latency_fl.item(), zc_fl.item())

            date_time = datetime.now().strftime('%m/%d %I:%M:%S %p') + ' | '
            pbar.set_description(date_time + s)

    # print(f"flops_acc: {flops_acc}, latency_acc: {latency_acc}, zc_acc: {zc_acc}")       
    ##############################################################
    # Print Continuous FLOP Value
    ##############################################################
    arch_prob = model.module.softmax_sampling(temperature) if is_ddp else model.softmax_sampling(temperature)
    architecture_info = {
        'arch_type': 'continuous',
        'arch': arch_prob
    }
    output_flops  = model.calculate_flops_new(architecture_info, est.flops_dict) / 1e3
    output_params = model.calculate_params_new(architecture_info, est.params_dict)
    output_depth = nn_model.calculate_layers_new(architecture_info)
    output_latency = model.calculate_table_latency(architecture_info, table)
    zc_score = model.calculate_zc(architecture_info, zc_map)
    # print(f'Continuous Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   Latency: {output_latency:.2f}ms   ZC: {zc_score}')
    print(f'Continuous Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   Depth: {output_depth}   Latency: {output_latency:.2f}ms   ZC: {zc_score}')
    
    ##############################################################
    # Print Discrete FLOP Value
    ##############################################################
    arch_prob = model.module.discretize_sampling() if is_ddp else model.discretize_sampling()
    architecture_info = {
        'arch_type': 'continuous',
        'arch': arch_prob
    }
    output_flops  = model.calculate_flops_new (architecture_info, est.flops_dict) / 1e3
    output_params = model.calculate_params_new(architecture_info, est.params_dict)
    output_depth = nn_model.calculate_layers_new(architecture_info)
    output_latency = model.calculate_table_latency(architecture_info, table)
    zc_score = model.calculate_zc(architecture_info, zc_map)
    # print(f'Discrete Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   Latency: {output_latency:.2f}ms   ZC: {zc_score}')
    print(f'Discrete Current FLOPS: {output_flops:.2f}G   Params: {output_params:.2f}M   Depth: {output_depth}   Latency: {output_latency:.2f}ms   ZC: {zc_score}')
            
    logger.info(s)
    logger.info(uw)
    return nn_model.thetas_main
    
        