#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:55:56 2018

@author: lz
"""

import os
import os.path as osp
import numpy as np
import pickle

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from datareader import Market1501Reader, CUHK03Reader, DukeMTMCReader
from datareader import grab_batch, grab_triplet_batch, concat_images
from models import ResNet, Discriminator
from utils import adjust_lr, set_lr, Hamming_dist, Euclidean_dist

import argparse
import time

seed = 712# 12 july is the date when I start this project. :)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

''' =========================== Configurations ========================== '''
parser = argparse.ArgumentParser()
parser.add_argument('--hash_bit', required=True, type=int)
parser.add_argument('--trial', type=int, default=1)
parser.add_argument('--dataset', type=str, default='dukemtmc-reid')
parser.add_argument('--source', type=str, default='detected',
                    help="Only for CUHK03, detected or labelled.")
parser.add_argument('--data_dir', type=str, default='../data/DukeMTMC-reID',
                    help="The root directory of data. Supports datasets: CUHK03, Market-1501, DukeMTMC-reID.")
parser.add_argument('--model_dir', type=str, default='../models')
parser.add_argument('--result_dir', type=str, default='../results')
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--resize_h', type=int, default=-1,
                    help="The h of input images. -1 represent no resize.")
parser.add_argument('--resize_w', type=int, default=-1,
                    help="The w of input images. -1 represent no resize.")
parser.add_argument('--transpose_axis', type=bool, default=True,
                    help="if Ture, (h, w, ch) --> (ch, h, w)")
parser.add_argument('--finetune_n_iters', type=int, default=2000)
parser.add_argument('--finetune_lr', type=float, default=0.1)
parser.add_argument('--finetune_batch_size', type=int, default=64)
parser.add_argument('--triplet_n_iters', type=int, default=16000)
parser.add_argument('--triplet_init_lr', type=float, default=0.001)
parser.add_argument('--triplet_batch_size', type=int, default=64)
parser.add_argument('--show_iters', type=int, default=100)
parser.add_argument('--D_lr', type=float, default=0.01)
parser.add_argument('--gan_opti_stride', type=int, default=40)
parser.add_argument('--n_gan_iters', type=int, default=20)
parser.add_argument('--disc_iter', type=int, default=5)
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--gpu_id', type=str, default='0')
args = parser.parse_args()
''' =========================== Other settings ============================ '''
if args.dataset == 'cuhk03':
    dtype = np.float32### for saving memory ###
    augmentation={'mirror':             'horizontal',#None,#
                  'crop':               None,#['topleft', 'topright', 'bottomleft', 'bottomright'],
                  'crop_size':          None,
                  'value_change':       None,
                  'saturation_change':  None}
    triplet_lrs = {2000: 0.0005, 3000: 0.00025, 5000: 0.0001}
    triplet_margins = {0: 0.2, 1000: 0.3, 2500: 0.4, 4000: 0.5}
elif args.dataset == 'market1501':
    dtype = np.float32### for saving memory ###
    augmentation={'mirror':             'horizontal',#None,#
                  'crop':               None,
                  'crop_size':          None,
                  'value_change':       None,
                  'saturation_change':  None}
    triplet_lrs = {4000: 0.0005, 6000: 0.00025, 10000: 0.0001}
    triplet_margins = {0: 0.2, 2000: 0.3, 6000: 0.4, 10000: 0.5}
elif args.dataset == 'dukemtmc-reid':
    dtype = np.float32### for saving memory ###
    augmentation={'mirror':             'horizontal',#None,#
                  'crop':               None,
                  'crop_size':          None,
                  'value_change':       None,
                  'saturation_change':  None}
    triplet_lrs = {4000: 0.0005, 6000: 0.00025, 10000: 0.0001}
    triplet_margins = {0: 0.2, 4000: 0.3, 10000: 0.4}
''' ============================ Get ready ========================== '''

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
cudnn.benchmark = True

### make directories ###
if not osp.exists(args.model_dir):
    os.makedirs(args.model_dir)
if not osp.exists(args.result_dir):
    os.makedirs(args.result_dir)

hash_bit = args.hash_bit
if hash_bit == 2048:
    num_features = 0
else:
    num_features = hash_bit

if args.resize_h != -1 and args.resize_w != -1:
    re_size = (args.resize_h, args.resize_w)
else:
    re_size = None

### normalization factor lambda ###
expectation_bernoulli = 0.5
binary_norm = 1 / np.sqrt( hash_bit * (0.5**2) )
#binary_norm = float('%.3f'%(binary_norm))

''' =========================== Load data ============================= '''
if args.dataset == 'cuhk03':
    datareader = CUHK03Reader(args.data_dir, re_size, args.result_dir, dtype,
                        args.transpose_axis, augmentation)
    datareader.get_data_info(args.source, args.trial)
    prbX, galX, prbY, galY = datareader.read_pair_images('train')
    X, Y, person_cam_index = concat_images(prbX, galX, prbY, galY)
    X_n = X.shape[0]
    del prbX, galX, prbY, galY
elif args.dataset == 'market1501':
    datareader = Market1501Reader(args.data_dir, re_size, args.result_dir, dtype,
                            args.transpose_axis, augmentation)
    X, Y, person_cam_index = datareader.read_images('train',
                                                       need_augmentation=True)
    X_n = X.shape[0]
elif args.dataset == 'dukemtmc-reid':
    datareader = DukeMTMCReader(args.data_dir, re_size, args.result_dir, dtype,
                        args.transpose_axis, augmentation)
    datareader.get_data_info()
    X, Y, person_cam_index = datareader.read_images('train',
                                                        need_augmentation=True)
    X_n = X.shape[0]

if args.train:
    ''' =========================== Load model ============================= '''
    print('Loading Model.')
    model = ResNet(depth=50, pretrained=True, cut_at_pooling=False,
                     num_features=num_features, norm=False, dropout=0.5,
                     num_classes=datareader.num_class)
    model.cuda()
    
    ''' ===================== Fine-tune model by ID classification ============ '''
    n_iters = args.finetune_n_iters
    lr = args.finetune_lr
    batch_size = args.finetune_batch_size
    show_iters = args.show_iters
    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()
    # Optimizer
    if hasattr(model, 'base'):
        base_param_ids = set(map(id, model.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.SGD(param_groups, lr=lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
    print('Start Training.')
    loss_record = 0.
    acc_record = 0.
    best_acc = 0.
    bat_start = 0
    for it in range(n_iters):
        print('\rExtractor | iter %05d'%(it+1), end='')
        adjust_lr(it, lr, optimizer, lr_step_size=1000)
    
        ''' Grab a batch from X and Y. '''
        batch, label, bat_start = grab_batch(X, Y, bat_start, batch_size)
    
        x = Variable(torch.from_numpy(batch.astype(float)).float().cuda())
        y = Variable(torch.from_numpy(label).long().cuda())
        
        ''' Feedforward and Backward. '''
        outputs = model(x)
        loss = criterion(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record += loss.data.cpu().numpy()[0]
        oo = outputs.data.cpu().numpy()
        oc = np.argmax(oo, axis=1)
        acc = np.sum(oc == label) / batch_size
        acc_record += acc
    
        ''' Show loss and accuracy, save best model. '''
        if it > 0 and (it+1) % args.show_iters == 0:
            loss_record /= args.show_iters
            acc_record /= args.show_iters
            print('\rExtractor | iter %05d, average batch loss: %.5f, average batch accuracy: %.3f'%(it+1, loss_record, acc_record))
    
            if acc_record > best_acc:
                best_acc = acc_record
                torch.save(model.state_dict(),
                            osp.join(args.model_dir, 'best_extractor_%d.pth'%(hash_bit)))
            loss_record = 0.
            acc_record = 0.
    
            if best_acc > 0.98:
                print('Extractor | Early Stop Training.')
                break
    
    del criterion
    del optimizer
    
    ''' ==================== Triplet Similarity Learning ==================== '''
    model.norm = True
    model.zero_grad()
    
    batch_size = args.triplet_batch_size
    n_iters = args.triplet_n_iters
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.triplet_init_lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
    
    one = torch.FloatTensor([1]).cuda()
    mone = one * -1
    D = Discriminator(hash_bit)
    D.cuda()
    D_solver = torch.optim.SGD(D.parameters(), lr=args.D_lr,
                                momentum=0.9,
                                weight_decay=5e-4,
                                nesterov=True)
    
    print('Start Triplet Similarity Training.')
    errors_real_D = []
    errors_fake_D = []
    errors_D = []
    errors_G = []
    losses_triplet = []
    loss_record = 0.
    best_loss = 10000.
    margin_change_count = 0
    gen_iter = 0
    bat_start = 0
    for it in range(n_iters):
        print('\rTriplet iter %05d'%(it+1), end='')
        if it in triplet_lrs.keys():
            set_lr(triplet_lrs[it], optimizer)
        if it in triplet_margins.keys():
            margin = triplet_margins[it]
        
        ''' Triplet Similarity Mearsuring '''
        anchor, positive, negative = grab_triplet_batch(X, Y,
                                                        person_cam_index,
                                                        datareader.num_class,
                                                        batch_size)
    
        x = Variable(torch.from_numpy(anchor.astype(float)).float().cuda())
        anc_feat = model.extract(x)
        
        x = Variable(torch.from_numpy(positive.astype(float)).float().cuda())
        pos_feat = model.extract(x)
        
        x = Variable(torch.from_numpy(negative.astype(float)).float().cuda())
        neg_feat = model.extract(x)
    
        loss = F.triplet_margin_loss(anc_feat, pos_feat, neg_feat,
                                     margin, p=2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_record += loss.data.cpu().numpy()[0]
        losses_triplet.append(loss.data.cpu().numpy()[0])
        
        del x, anc_feat, pos_feat, neg_feat
        del loss
        
        ''' Hashing Regularization '''
        if it > 0 and it % args.gan_opti_stride == 0:
            print('')
            for regular_it in range(args.n_gan_iters):
                print('\rHashing iter %05d Hashing Regularization %03d / %03d'%(it+1,
                                                                        regular_it+1,
                                                                        args.n_gan_iters),
                      end='')
                ############################
                # (1) Update D network
                ###########################
                for p in D.parameters(): # reset requires_grad
                    p.requires_grad = True # they are set to False below in netG update
                if gen_iter < 10 or gen_iter % 500 == 0:
                    disc_update_iter = 10
                else:
                    disc_update_iter = args.disc_iter
                for di in range(disc_update_iter):
    #                print('\rUpdating Discriminator %03d / %03d'%(di+1, disc_update_iter), end='')
                    ''' Descriminator: clamp parameters to a cube '''
                    for p in D.parameters():
                        p.data.clamp_(args.clamp_lower, args.clamp_upper)
                    ''' Grab a batch '''
                    batch, label, bat_start = grab_batch(X, Y, bat_start, batch_size)
                    x = Variable(torch.from_numpy(batch.astype(float)).float().cuda())
                    feat = model.extract(x)
                    
                    binary_code = torch.bernoulli(torch.rand(batch_size, hash_bit)).cuda()
                    binary_code *= binary_norm
                    bc = Variable(binary_code)
                    
                    D_solver.zero_grad()
                    
                    errD_real = D(bc)
                    errD_real.backward(one)
                    errors_real_D.append(errD_real.data.cpu().numpy()[0])
                    
                    errD_fake = D(feat)
                    errD_fake.backward(mone)
                    errors_fake_D.append(errD_fake.data.cpu().numpy()[0])
                    
                    errD = errD_real - errD_fake
                    errors_D.append(errD.data.cpu().numpy()[0])
                    D_solver.step()
                    
                    del x, feat
                    del errD_real, errD_fake, errD
                ############################
                # (2) Update G network
                ###########################
                for p in D.parameters():
                    p.requires_grad = False # to avoid computation
                model.zero_grad()
                optimizer.zero_grad()
                gan_batch, label, bat_start = grab_batch(X, Y, bat_start, batch_size)
                x = Variable(torch.from_numpy(gan_batch.astype(float)).float().cuda())
                feat = model.extract(x)
                errG = D(feat)
                errors_G.append(errG.data.cpu().numpy()[0])
                errG.backward(one)
                optimizer.step()
                gen_iter += 1
                
                del x, feat
                del errG
    
        ''' Show Training Info '''
        if it > 0 and (it+1) % args.show_iters == 0:
            loss_record /= args.show_iters
            print('\rTriplet iter %05d, average batch loss: %.8f, current margin %f'%(it+1, loss_record, margin))
    
            if loss_record < best_loss:
                best_loss = loss_record
                torch.save(model.state_dict(),
                            osp.join(args.model_dir, 'best_triplet_%d.pth'%(hash_bit)))
    
            loss_record = 0.
    
    with open(osp.join(args.result_dir, 'train_logs_%d_%d.pkl'%(hash_bit, args.trial)), 'wb') as f:
        pickle.dump({'settings':{'init_lr': args.triplet_init_lr,
                                 'lrs': triplet_lrs,
                                 'batch_size': args.triplet_batch_size,
                                 'n_iters': args.triplet_n_iters,
                                 'margins': triplet_margins,
                                 'D_lr': args.D_lr,
                                 'regularization_iter': args.gan_opti_stride,
                                 'n_regularization_iters': args.n_gan_iters,
                                 'disc_iter': args.disc_iter
                                 },
                     'errors_real_D': errors_real_D,
                     'errors_fake_D': errors_fake_D,
                     'errors_D': errors_D,
                     'errors_G': errors_G,
                     'losses_triplet': losses_triplet
                     }, f)
else:
    print('Loading Existing Model.')
    model = ResNet(depth=50, pretrained=False, cut_at_pooling=False,
                     num_features=num_features, norm=False, dropout=0.5,
                     num_classes=datareader.num_class)
    model.load_state_dict(torch.load(osp.join(args.model_dir, 'best_triplet_%d.pth'%(hash_bit))))
    model.cuda()

''' ------------------------------- Testing --------------------------------- '''
if args.test:
    batch_size = args.triplet_batch_size
    ''' ============================= Testing ================================ '''
    model.eval()
    n_feat = hash_bit
    ''' Testing Query Features '''
    if args.dataset == 'cuhk03':
        prbX, galX, prbY, galY = datareader.read_pair_images('test', need_augmentation=False)
    elif args.dataset == 'market1501':
        prbX, prbY, _ = datareader.read_images('query', need_augmentation=False, need_shuffle=False)
        galX, galY, _ = datareader.read_images('test', need_augmentation=False,
                                                         include_distractors=True, need_shuffle=False)
    elif args.dataset == 'dukemtmc-reid':
        prbX, prbY, _ = datareader.read_images('query', need_augmentation=False, need_shuffle=False)
        galX, galY, _ = datareader.read_images('test', need_augmentation=False, need_shuffle=False)
    prbFeatX = np.zeros((prbX.shape[0], n_feat), dtype=dtype)
    bn = int(np.ceil(prbX.shape[0] / batch_size))
    bi = 0
    bc = 0
    times = []
    while bi < prbX.shape[0]:
        bc += 1
        print('\rExtractor | Extract testing query feature %04d / %04d batch'%(bc, bn), end='')
        be = min(bi + batch_size, prbX.shape[0])
        batch = prbX[bi:be, ...]
        x = Variable(torch.from_numpy(batch.astype(float)).float().cuda())
        t0 = time.time()
        feat = model.extract(x)
        t1 = time.time()
        times.append(t1-t0)
        prbFeatX[bi:be, ...] = feat.data.cpu().numpy()
        bi = be
    print(' ... Done.')
    time_per_sample = np.sum(times) / prbX.shape[0]
    del prbX
    
    ''' Testing Gallery Features '''
    galFeatX = np.zeros((galX.shape[0], n_feat), dtype=dtype)
    bn = int(np.ceil(galX.shape[0] / batch_size))
    bi = 0
    bc = 0
    while bi < galX.shape[0]:
        bc += 1
        print('\rExtractor | Extract testing gallery feature %04d / %04d batch'%(bc, bn), end='')
        be = min(bi + batch_size, galX.shape[0])
        batch = galX[bi:be, ...]
        x = Variable(torch.from_numpy(batch.astype(float)).float().cuda())
        feat = model.extract(x)
        galFeatX[bi:be, ...] = feat.data.cpu().numpy()
        bi = be
    print(' ... Done.')
    del galX
    
    print('Computing Euclidean distances between features.')
    dist0, rank0, cmc0, mAP0, euclid_dist_time = Euclidean_dist(prbFeatX, galFeatX, prbY, galY, calc_mAP=True)
    
    print('Computing Hashing distances.')
    prbHash = np.zeros_like(prbFeatX)
    galHash = np.zeros_like(galFeatX)
    prbHash[prbFeatX>=binary_norm/2] = 1
    galHash[galFeatX>=binary_norm/2] = 1
    
    dist, rank, cmc, mAP, hamming_dist_time = Hamming_dist(prbHash, galHash, prbY, galY, calc_mAP=True)
    
    with open(osp.join(args.result_dir, 'test_logs_%d_%d.pkl'%(hash_bit, args.trial)), 'wb') as f:
        pickle.dump({'time_extract_per_sample': time_per_sample,
                     'cmc_float': cmc0,
                     'mAP_float': mAP0,
                     'cmc_binary': cmc,
                     'mAP_binary': mAP,
                     'euclid_dist_time': euclid_dist_time,
                     'hamming_dist_time': hamming_dist_time
                     }, f)








