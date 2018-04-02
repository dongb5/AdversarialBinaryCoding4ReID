#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 16:07:02 2018

@author: lz
"""

import time
import numpy as np
from sklearn.metrics import average_precision_score

def adjust_lr(epoch, lr, optimizer, lr_step_size=200):
    lr = lr * (0.6 ** (epoch // lr_step_size))
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

def set_lr(lr, optimizer):
    for g in optimizer.param_groups:
        g['lr'] = lr * g.get('lr_mult', 1)

def Hamming_dist(prb, gal, prb_labels, gal_labels, calc_mAP=False):
    if calc_mAP:
        mAP = 0.
    else:
        mAP = None
    pn = prb.shape[0]
    gn = gal.shape[0]
    
    t = 0
    dist = np.zeros((pn, gn))
    for i in range(pn):
        print('\rHamming Distance %05d / %05d'%(i+1, pn), end='')
        pp = np.tile(prb[i, :], (gn, 1))
        t0 = time.time()
        s = np.logical_xor(pp, gal)
        t += time.time() - t0
        dist[i, :] = np.sum( s, axis=1 )
    t /= pn
    print(' ... Done.')
    
    rank = np.argsort(dist, axis=1)
    cmc = np.zeros((rank.shape[1]))
    for pi in range(pn):
        idx = -1
        for gi in range(gn):
            if prb_labels[pi] == gal_labels[rank[pi, gi]]:
                idx = gi
                break
        if idx >= 0:
            cmc[idx:] += 1
        if calc_mAP:
            y_true = np.zeros((gn))
            y_true[gal_labels == prb_labels[pi]] = 1
            mAP += average_precision_score(y_true, -dist[pi, :])
    cmc /= pn
    if calc_mAP:
        mAP /= pn
    
    return dist, rank, cmc, mAP, t

def Euclidean_dist(prb, gal, prb_labels, gal_labels, calc_mAP=False):
    if calc_mAP:
        mAP = 0.
    else:
        mAP = None
    pn = prb.shape[0]
    gn = gal.shape[0]
    
    t = 0
    dist = np.zeros((pn, gn))
    for i in range(pn):
        t0 = time.time()
        for j in range(gn):
            dist[i, j] = np.sqrt(np.sum((prb[i,:]-gal[j,:])**2))
        t += time.time() - t0
    t /= pn
    
    rank = np.argsort(dist, axis=1)
    cmc = np.zeros((rank.shape[1]))
    for pi in range(pn):
        idx = -1
        for gi in range(gn):
            if prb_labels[pi] == gal_labels[rank[pi, gi]]:
                idx = gi
                break
        if idx >= 0:
            cmc[idx:] += 1
        if calc_mAP:
            y_true = np.zeros((gn))
            y_true[gal_labels == prb_labels[pi]] = 1
            mAP += average_precision_score(y_true, -dist[pi, :])
    cmc /= pn
    if calc_mAP:
        mAP /= pn
    
    return dist, rank, cmc, mAP, t

def rank_cmc_mAP(dist, prb_labels, gal_labels, calc_mAP=False):
    if calc_mAP:
        mAP = 0.
    else:
        mAP = None
    
    pn, gn = dist.shape
    
    rank = np.argsort(dist, axis=1)
    cmc = np.zeros((rank.shape[1]))
    for pi in range(pn):
        idx = -1
        for gi in range(gn):
            if prb_labels[pi] == gal_labels[rank[pi, gi]]:
                idx = gi
                break
        if idx >= 0:
            cmc[idx:] += 1
        if calc_mAP:
            y_true = np.zeros((gn))
            y_true[gal_labels == prb_labels[pi]] = 1
            mAP += average_precision_score(y_true, -dist[pi, :])
    cmc /= pn
    if calc_mAP:
        mAP /= pn
    
    return rank, cmc, mAP