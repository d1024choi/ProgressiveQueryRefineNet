# from utils.libraries import *
import json
import os
import glob
import sys
import numpy as np
import shutil
import pickle
from pathlib import Path
import cv2
import time
from tqdm import tqdm
import logging
import traceback
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def read_config(model_name, path=None):

    if (path is None):
        cfg = read_json(path='./config/config.json')
        cfg.update(read_json(path=f'./config/{model_name}/data.json'))
        cfg.update(read_json(path=f'./config/{model_name}/loss.json'))
        cfg.update(read_json(path=f'./config/{model_name}/model.json'))
    else:
        file_path = os.path.join(path, 'config.json')
        cfg = read_json(path=file_path)

        file_path = os.path.join(path, f'{model_name}/data.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'{model_name}/loss.json')
        cfg.update(read_json(path=file_path))

        file_path = os.path.join(path, f'{model_name}/model.json')
        cfg.update(read_json(path=file_path))

    cfg['nuscenes']['dataset_dir'] = check_dataset_path(cfg['nuscenes']['dataset_dir'])

    return cfg

def config_update(cfg, args):

    # --------------------------------------------
    # from args to cfg
    # --------------------------------------------
    cfg['bev']['h'] = args.bev_h
    cfg['bev']['w'] = args.bev_w
    cfg['bev']['h_meters'] = args.bev_h_meters
    cfg['bev']['w_meters'] = args.bev_w_meters
    cfg['bev']['offset'] = args.bev_offset

    cfg['image']['top_crop'] = args.img_top_crop
    cfg['image']['h'] = args.img_h
    cfg['image']['w'] = args.img_w

    update = {'training_params': {'exp_id': args.exp_id,
                                  'num_epochs': args.num_epochs,
                                  'batch_size': args.batch_size,
                                  'learing_rate': args.learning_rate,
                                  'weight_decay': args.weight_decay,
                                  'min_learing_rate': args.min_learning_rate,
                                  'grad_clip': args.grad_clip,
                                  'weights': {'vehicle': args.w_vehicle,
                                              'pedestrian': args.w_pedestrian,
                                              'road': args.w_road,
                                              'lane': args.w_lane,
                                              'intm': args.w_intm,
                                              'offset': args.w_offset},
                                  'apply_lr_scheduling': args.apply_lr_scheduling
                                  },
              'img_augmentation' : {'max_rotation': args.max_rotation,
                                    'max_translation': args.max_translation,
                                    'max_scale_severity': args.max_scale_severity},
              'obs_len': args.obs_len,
              'pred_len': args.pred_len,
              'bool_mixed_precision': bool(args.bool_mixed_precision),
              'img_aug_prob': args.img_aug_prob
              }
    cfg.update(update)

    # --------------------------------------------
    # Scratch
    # --------------------------------------------
    if (args.model_name == 'Scratch'):

        if (args.target != 'none'):
            cfg['target'] = [args.target]

        cfg['encoder']['repeat'] = args.num_enc_repeat
        update = {'bool_use_vis_offset': bool(args.bool_use_vis_offset),
                  'target_feat_levels': args.target_feat_levels,
                  # 'feat_int_method': args.feat_inter_method,
                  'feat_int_repeat': args.feat_inter_repeat,
                  'decoder_type': args.decoder_type,
                  # 'cross_attn_method': args.cross_attn_method,
                  'hierarchy_depth': args.hierarchy_depth,
                  'bool_dec_head': bool(args.bool_add_dec_header),
                  'bool_learn_offset': bool(args.bool_learn_offset),
                  'bool_apply_crosshead': bool(args.bool_apply_crosshead),
                  'bool_save_attn_info' : bool(args.bool_save_attn_info),
                  'ddp': bool(args.ddp)}
        cfg.update(update)
    return cfg

def check_dataset_path(path, servers=['dooseop', 'etri']):

    if (os.path.exists(path)):
        return path
    else:
        current_machine = None
        for server in servers:
            if (path.find(server) > 0):
                current_machine = server
                break

        for server in servers:
            if (os.path.exists(path.replace(current_machine, server))):
                return path.replace(current_machine, server)

    sys.exit('>> Unable to locate dataset path..')

def get_dtypes(useGPU=True):
    return torch.LongTensor, torch.FloatTensor

def init_weights(m):

    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)

def toNP(x):

    return x.detach().to('cpu').numpy()

def toTS(x, dtype):

    return torch.from_numpy(x).to(dtype)

def save_read_latest_checkpoint_num(path, val, isSave):

    file_name = path + '/checkpoint.txt'
    index = 0

    if (isSave):
        file = open(file_name, "w")
        file.write(str(int(val)) + '\n')
        file.close()
    else:
        if (os.path.exists(file_name) == False):
            print('[Error] there is no such file in the directory')
            sys.exit()
        else:
            f = open(file_name, 'r')
            line = f.readline()
            index = int(line[:line.find('\n')])
            f.close()

    return index

def read_all_saved_param_idx(path):
    ckp_idx_list = []
    files = sorted(glob.glob(os.path.join(path, '*.pt')))
    for i, file_name in enumerate(files):
        start_idx = 0
        for j in range(-3, -10, -1):
            if (file_name[j] == '_'):
                start_idx = j+1
                break
        ckp_idx = int(file_name[start_idx:-3])
        ckp_idx_list.append(ckp_idx)
    return ckp_idx_list[::-1]

def copy_chkpt_every_N_epoch(args):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index

    root_path = args.model_dir + str(args.exp_id)
    save_directory =  root_path + '/copies'
    if save_directory != '' and not os.path.exists(save_directory):
        os.makedirs(save_directory)

    fname_list = []
    fnum_list = []
    all_file_names = os.listdir(root_path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    max_idx = np.argmax(np.array(fnum_list))
    target_file = fname_list[max_idx]

    src = root_path + '/' + target_file
    dst = save_directory + '/' + target_file
    shutil.copy2(src, dst)

    print(">> {%s} is copied to {%s}" % (target_file, save_directory))

def remove_past_checkpoint(path, max_num=5):

    def get_file_number(fname):

        # read checkpoint index
        for i in range(len(fname) - 3, 0, -1):
            if (fname[i] != '_'):
                continue
            index = int(fname[i + 1:len(fname) - 3])
            return index


    num_remain = max_num - 1
    fname_list = []
    fnum_list = []

    all_file_names = os.listdir(path)
    for fname in all_file_names:
        if "saved" in fname:
            chk_index = get_file_number(fname)
            fname_list.append(fname)
            fnum_list.append(chk_index)

    if (len(fname_list)>num_remain):
        sort_results = np.argsort(np.array(fnum_list))
        for i in range(len(fname_list)-num_remain):
            del_file_name = fname_list[sort_results[i]]
            os.remove('./' + path + '/' + del_file_name)

def print_current_train_progress(e, b, num_batchs, time_spent, total_loss):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r [Epoch %02d] %d / %d (%.4f sec/sample), total loss : %.4f' % (e, b, num_batchs, time_spent, total_loss)),

    sys.stdout.flush()

def print_current_valid_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> validation process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_current_test_progress(b, num_batchs):

    if b >= num_batchs-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\r >> test process (%d / %d) ' % (b, num_batchs)),

    sys.stdout.flush()

def print_training_info(args, logger):


    logger.info("--------- %s / %s ----------" % (args.dataset_type, args.model_name))
    logger.info(" Exp id : %d" % args.exp_id)
    logger.info(" Gpu num : %d" % args.gpu_num)
    logger.info(" Num epoch : %d" % args.num_epochs)
    logger.info(" Batch size : %d" % args.batch_size)
    logger.info(" Data Seq. length : %d" % (args.obs_len + args.pred_len))
    logger.info(" Mixed preicision : %d " % (args.bool_mixed_precision))
    logger.info(" weights for veh/ped/road/lane/intm/offset : %.4f/%.4f/%.4f/%.4f/%.4f/%.4f"
                % (args.w_vehicle, args.w_pedestrian, args.w_road, args.w_lane, args.w_intm, args.w_offset))
    logger.info(" initial learning rate/weight decay : %.5f/%.7f " % (args.learning_rate, args.weight_decay))
    logger.info("----------------------------------")

    if (args.apply_lr_scheduling == 1):
        logger.info(" * One Cycle LR Scheduler")
        logger.info(" div factor : %.2f" % args.div_factor)
        logger.info(" pct start : %.2f" % args.pct_start)
        logger.info(" final div factor : %.2f" % args.final_div_factor)
    logger.info("----------------------------------")

    logger.info(" Image Augmentation Prob : %.2f" % args.img_aug_prob)
    logger.info(" Rotation (%.2f), Translation (%.2f), Scaling (%.2f) " % (args.max_rotation, args.max_translation, args.max_scale_severity))
    logger.info("----------------------------------")

    logger.info(" # of MR query maps : %d" % args.hierarchy_depth)
    logger.info(" Target Img. Feat. Map Levels : %s" % (args.target_feat_levels))
    logger.info(" Headers for Decoder : %d" % args.bool_add_dec_header)
    logger.info(" Decoder type : %s" % args.decoder_type)
    logger.info(" Learn offset (w/ visibility map %d) : %d" % (args.bool_use_vis_offset, args.bool_learn_offset))
    logger.info(" Cross task interaction : %d" % args.bool_apply_crosshead)
    logger.info("----------------------------------")
