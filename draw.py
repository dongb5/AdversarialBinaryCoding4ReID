#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 17:15:20 2018

@author: lz
"""

import os
import pickle
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_file', type=str, required=True,)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--fig_dir', type=str, default='./figs')
parser.add_argument('--savefmt', type=str, default='pdf')
args = parser.parse_args()

if not osp.exists(args.fig_dir):
    os.makedirs(args.fig_dir)

with open(args.log_file, 'rb') as f:
    logs = pickle.load(f)

losses_triplet = logs['losses_triplet']
errors_real_D = logs['errors_real_D']
errors_fake_D = logs['errors_fake_D']
errors_D = logs['errors_D']
errors_G = logs['errors_G']

margins = logs['settings']['margins']
margin_iter = list(margins.keys())

titles = {'market1501': 'Market-1501',
          'dukemtmc-reid': 'DukeMTMC-reID',
          'cuhk03labeled': 'CUHK03 (labelled)',
          'cuhk03detected': 'CUHK03 (detected)',}

plt.close('all')
    
### Triplet loss
plt.figure(figsize=(6,4))
plt.grid(True)
plt.title("Triplet loss on "+titles[args.dataset], fontsize=22)
ltri_stride = int(len(losses_triplet)/100)
ltri = np.mean(np.reshape(losses_triplet, [100, ltri_stride]), axis=1)
x = range(1, len(losses_triplet)+1, ltri_stride)
plt.plot(x, ltri, linewidth=5, color='r')

plt.ylabel('Loss', fontsize=20)
plt.xlabel('Iteration', fontsize=20)
plt.legend(['Triplet loss'], fontsize=16)

for dash in range(1, len(margin_iter)):
    plt.axvline(x=margin_iter[dash], ymin=0, ymax=1, linewidth=3, linestyle='dashed', color='k')

for anno in margin_iter:
    plt.text(x=anno+100, y=0.08, s='margin\n = %.1f'%(margins[anno]), fontsize=18,
         verticalalignment='bottom', horizontalalignment='left')

plt.savefig(osp.join(args.fig_dir, 'triplet_loss.%s'%(args.savefmt)),
                    format=args.savefmt, bbox_inches='tight')

### D loss
plt.figure(figsize=(6,4))
plt.grid(True)
plt.title("Loss of Discriminator on "+titles[args.dataset], fontsize=19)
x = range(len(errors_real_D))
plt.plot(x, errors_D, linewidth=5)
plt.plot(x, errors_real_D, linewidth=5)
plt.plot(x, errors_fake_D, linewidth=5)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Discriminator iteration', fontsize=20)
plt.legend(['Loss of Discriminator',
            'Loss of D on binary vectors',
            'Loss of D on features'],
           fontsize=12, loc='lower left')

plt.savefig(osp.join(args.fig_dir, 'D_loss.%s'%(args.savefmt)),
            format=args.savefmt, bbox_inches='tight')

### G loss
plt.figure(figsize=(6,4))
plt.grid(True)
plt.title("Loss of Generator on "+titles[args.dataset], fontsize=19)
x = range(len(errors_G))
plt.plot(x, errors_G, linewidth=5)
plt.ylabel('Loss', fontsize=20)
plt.xlabel('Generator iteration', fontsize=20)
plt.legend(['Loss of Generator'], fontsize=18)

plt.savefig(osp.join(args.fig_dir, 'G_loss.%s'%(args.savefmt)),
            format=args.savefmt, bbox_inches='tight')



