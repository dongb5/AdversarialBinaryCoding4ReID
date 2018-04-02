#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:57:22 2018

@author: lz
"""

from __future__ import absolute_import

import numpy as np
import os
from os.path import join as pathjoin
import pickle

import skimage
import skimage.io as skio
import skimage.transform as sktr
import skimage.color as skcolor
import glob
import random



class ImageReader(object):
    def __init__(self, data_dir, re_size=None, dtype=np.float32, transpose_axis=True,
                 augmentation={'mirror':None,
                               'crop':None,
                               'crop_size':None,
                               'value_change':None,
                               'saturation_change':None}):
        self.data_dir = data_dir
        self.re_size = re_size
        self.dtype = dtype
        self.transpose_axis = transpose_axis
        self.augmentation = augmentation
        self.mean = None
        self.std = None
        
    def get_augmentation_num(self, augmentation):
        aug_num = 1
        if augmentation['mirror'] is not None:
            aug_num += 1
        if augmentation['crop'] is not None:
            aug_num += aug_num * len(augmentation['crop'])

        ratios = []
        if augmentation['value_change'] is not None\
        and augmentation['saturation_change'] is not None:
            for saturation_ratio in augmentation['saturation_change']:
                for value_ratio in augmentation['value_change']:
                    ratios.append( (saturation_ratio, value_ratio) )
        elif augmentation['saturation_change'] is not None:
            for saturation_ratio in augmentation['saturation_change']:
                ratios.append( (saturation_ratio, 1) )
        elif augmentation['value_change'] is not None:
            for value_ratio in augmentation['value_change']:
                ratios.append( (1, value_ratio) )
        if ratios != []:
            aug_num += aug_num * len(ratios)
        return aug_num
    
    def split_train_test(self):
        pass 
    
    def compute_training_mean_std(self):
        pass
    
    def read_pair_images(self, part, need_augmentation=True):
        pass
    
    def read_images(self, part, need_augmentation=True):
        pass
    
    def image_augmentation(self, img, augmentation):
        image_list = [img]
        crop_end = 1
        sv_end = 1
        ### mirror image
        if augmentation['mirror'] == 'horizontal':
            image_list.append(self.horizontal_mirror_image(img))
            crop_end += 1
            sv_end += 1
        ### crop images
        if augmentation['crop'] is not None:
            crop_size = augmentation['crop_size']
            for i in range(crop_end):#, im in enumerate(image_list[:crop_end]):
                im = image_list[i]
                for crop_mode in augmentation['crop']:
                    cropped_img = self.crop_image(im, crop_mode, crop_size)
                    image_list.append(cropped_img)
                    sv_end += 1
            for i, im in enumerate(image_list[:crop_end]):
                image_list[i] = sktr.resize(im, crop_size, mode='reflect')

        ### change light value and saturation
        if augmentation['value_change'] is not None\
        and augmentation['saturation_change'] is not None:
            ratios = []
            for saturation_ratio in augmentation['saturation_change']:
                for value_ratio in augmentation['value_change']:
                    ratios.append( (saturation_ratio, value_ratio) )
        elif augmentation['saturation_change'] is not None:
            ratios = []
            for saturation_ratio in augmentation['saturation_change']:
                ratios.append( (saturation_ratio, 1) )
        elif augmentation['value_change'] is not None:
            ratios = []
            for value_ratio in augmentation['value_change']:
                ratios.append( (1, value_ratio) )
        else:
            ratios = None

        if ratios is not None:
            for im in image_list[:sv_end]:
                for s, v in ratios:
                    image_list.append(self.hsv_image_change(im, s, v))
        return image_list

    def horizontal_mirror_image(self, img):
        w = img.shape[1]
        mirror_img = np.zeros_like(img)
        for x in range(int(w / 2)):
            mirror_img[:, x, :] = img[:, w-1-x, :]
            mirror_img[:, w-1-x, :] = img[:, x, :]
        return mirror_img

    def crop_image(self, img, mode, crop_size):
        '''
            mode: one of 'topleft', 'topright', 'bottomleft', 'bottomright',
                  'random'
        '''
        ch, cw = crop_size
        if mode == 'topleft':
            cropped_img = img[:ch, :cw, :]
        elif mode == 'topright':
            cropped_img = img[:ch, -cw:, :]
        elif mode == 'bottomleft':
            cropped_img = img[-ch:, :cw, :]
        elif mode == 'bottomright':
            cropped_img = img[-ch:, -cw:, :]
        elif mode == 'random':
            h, w, _ = img.shape
            cy = np.random.randint(0, h-ch)
            cx = np.random.randint(0, w-cw)
            cropped_img = img[cy:cy+ch, cx:cx+cw, :]
        return cropped_img

    def hsv_image_change(self, img, s, v):
        changed_image = self.hsv_image_saturation_change(img, s)
        changed_image = self.hsv_image_value_change(changed_image, v)
        return changed_image

    def hsv_image_value_change(self, img, ratio):
        hsvimg = skcolor.rgb2hsv(img)
        hsvimg[..., 2] *= ratio
        changed_image = (skcolor.hsv2rgb(hsvimg) * 255).astype(np.uint8)
        return changed_image

    def hsv_image_saturation_change(self, img, ratio):
        hsvimg = skcolor.rgb2hsv(img)
        hsvimg[..., 1] *= ratio
        changed_image = (skcolor.hsv2rgb(hsvimg) * 255).astype(np.uint8)
        return changed_image


class CUHK03Reader(ImageReader):
    ich = 3
    def __init__(self, data_dir, re_size, save_dir=None, dtype=np.float32,
                 transpose_axis=True,
                 augmentation={'mirror':None,
                               'crop':None,
                               'crop_size':None,
                               'value_change':None,
                               'saturation_change':None}):
        super(CUHK03Reader, self).__init__(data_dir, re_size, dtype, transpose_axis,
                                           augmentation)
        
        self.save_dir = save_dir
        
        self.labeled_dir = pathjoin(data_dir, 'labeled')
        self.detected_dir = pathjoin(data_dir, 'detected')
        with open(pathjoin(data_dir, 'testsets.pkl'), 'rb') as f:
            self.testsets = pickle.load(f)
        
        self.aug_num = self.get_augmentation_num(augmentation)
    
    def get_data_info(self, source, trial):
        ''' 
            source: 'labeled' or 'detected'.
            trial: int type.
        '''
        if self.save_dir is not None\
        and os.path.exists(pathjoin(self.save_dir, 'data_info_%s_trial%02d.pkl'%(source, trial))):
            print('CUHK03 Reader: Loading data info.')
            with open(pathjoin(self.save_dir, 'data_info_%s_trial%02d.pkl'%(source, trial)), 'rb') as f:
                info = pickle.load(f)
            self.train_files = info['train_files']
            self.test_files = info['test_files']
            self.mean = info['mean']
            self.std = info['std']
            self.min_h = info['min_h']
            self.min_w = info['min_w']
            self.max_h = info['max_h']
            self.max_w = info['max_w']
            self.num_class = info['num_class']
        else:
            print('CUHK03 Reader *** Warning ***: New data infomation will recover the old one.')
            self.mean = np.zeros((3), dtype=self.dtype)
            self.std = np.zeros((3), dtype=self.dtype)
            self.min_h = 1000
            self.min_w = 1000
            self.max_h = 0
            self.max_w = 0
            
            self.test_set = []
            for group, pid in self.testsets[trial]:
                self.test_set.append('group%d_%04d'%(group, pid))
            
            files = glob.glob(pathjoin(self.data_dir, source, '*.jpg'))
            files.sort()
            fn = len(files)
            
            self.train_files = {'cam1':[], 'cam2':[]}
            self.test_files = {'cam1':[], 'cam2':[]}
            self.train_pids = []
            tr_count = 0
            for fi, f in enumerate(files):
                print('\rCUHK03 Reader: read info from image %05d / %05d'%(fi+1, fn), end='')
                fname = f[f.rfind('/')+1:]
                img = skimage.img_as_float( skio.imread(f) )
                image = sktr.resize(img, self.re_size, mode='reflect')
                h, w, _ = img.shape
                if h < self.min_h:
                    self.min_h = h
                if h > self.max_h:
                    self.max_h = h
                if w < self.min_w:
                    self.min_w = w
                if w > self.max_w:
                    self.max_w = w
                if fname[:11] in self.test_set:
                    self.test_files[fname[12:16]].append(f)
                else:
                    pid = fname[7:11]
                    if pid not in self.train_pids:
                        self.train_pids.append(pid)
                    self.train_files[fname[12:16]].append(f)
                    self.mean[0] += np.mean(image[..., 0])
                    self.mean[1] += np.mean(image[..., 1])
                    self.mean[2] += np.mean(image[..., 2])
                    tr_count += 1
            self.mean /= tr_count
            print(' ... Done.')
            self.num_class = len(self.train_pids)
            
            if self.re_size is not None\
            and (self.min_h < self.re_size[0] or self.min_w < self.re_size):
                print('\nCUHK03 Reader *** Warning ***: Minimal image size smaller than reshape size.\n')
            
            for fi, f in enumerate(files):
                print('\rCUHK03 Reader: computing std %05d / %05d'%(fi+1, fn), end='')
                fname = f[f.rfind('/')+1:]
                if fname[:11] not in self.test_set:
                    img = skimage.img_as_float( skio.imread(f) )
                    img = sktr.resize(img, self.re_size, mode='reflect')
                    self.std[0] += np.mean( (img[..., 0] - self.mean[0])**2 )
                    self.std[1] += np.mean( (img[..., 1] - self.mean[1])**2 )
                    self.std[2] += np.mean( (img[..., 2] - self.mean[2])**2 )
            self.std /= tr_count
            self.std = np.sqrt(self.std)
            print(' ... Done.')
            
            if self.save_dir is not None:
                with open(pathjoin(self.save_dir, 'data_info_%s_trial%02d.pkl'%(source, trial)), 'wb') as f:
                    pickle.dump({'train_files': self.train_files,
                                 'test_files':  self.test_files,
                                 'mean':        self.mean,
                                 'std':         self.std,
                                 'min_h':       self.min_h,
                                 'min_w':       self.min_w,
                                 'max_h':       self.max_h,
                                 'max_w':       self.max_w,
                                 'num_class':   self.num_class}, f)
    
    def get_minimal_size(self):
        return self.min_h, self.min_w
    
    def set_reshape_size(self, re_size):
        self.re_size = re_size
    
    def set_minimal_size_as_reshape_size(self):
        self.re_size = (self.min_h, self.min_w)
    
    def read_pair_images(self, part, need_augmentation=True):
        '''
            Note: Get data infomation before read images
                  (run get_data_info(source, trial) function).
        '''
        if part == 'train':
            files = self.train_files
        elif part == 'test':
            files = self.test_files
        
        pn = len(files['cam1'])
        gn = len(files['cam2'])
        if self.augmentation['crop_size'] is not None:
            sz = self.augmentation['crop_size'] + (self.ich,)
        elif self.re_size is not None:
            sz = self.re_size + (self.ich,)
        else:
            sz = (self.ih, self.iw, self.ich)
        if self.transpose_axis:
            sz = (sz[2], sz[0], sz[1])
        
        if need_augmentation:
            aug_num = self.aug_num
        else:
            aug_num = 1
        
        prb_images = np.zeros((pn*aug_num,) + sz, dtype=self.dtype)
        gal_images = np.zeros((gn*aug_num,) + sz, dtype=self.dtype)
        prb_labels = np.zeros((pn*aug_num), dtype=int)
        gal_labels = np.zeros((gn*aug_num), dtype=int)
        
        pid_label = {}
        label = 0
        di = 0
        for fi, f in enumerate(files['cam1']):
            print('\rCUHK03 Reader: Reading %s prb images %04d / %04d'%(part, fi+1, pn), end='')
            fname = f[f.rfind('/')+1:]
            pid = fname[7:11]
            if pid not in pid_label.keys():
                pid_label[pid] = label
                label += 1
            
            image = skio.imread(f)
            if not self.dtype == np.uint8:
                image = skimage.img_as_float( image )
            image = sktr.resize(image, self.re_size, mode='reflect')
            if need_augmentation:
                img_list = self.image_augmentation(image, self.augmentation)
            else:
                img_list = [image]
            img_list = np.array(img_list, dtype=self.dtype)
            if self.mean is not None:
                img_list -= self.mean
            if self.std is not None:
                img_list /= self.std
            if self.transpose_axis:
                img_list = np.transpose(img_list, (0, 3, 1, 2))
            prb_images[di:di+aug_num, ...] = img_list
            prb_labels[di:di+aug_num] = pid_label[pid]
            di += aug_num
        print(' ... Done.')
        di = 0
        for fi, f in enumerate(files['cam2']):
            print('\rCUHK03 Reader: Reading %s gal images %04d / %04d'%(part, fi+1, gn), end='')
            fname = f[f.rfind('/')+1:]
            pid = fname[7:11]
            if pid not in pid_label.keys():
                pid_label[pid] = label
                label += 1
            
            image = skio.imread(f)
            if not self.dtype == np.uint8:
                image = skimage.img_as_float( image )
            image = sktr.resize(image, self.re_size, mode='reflect')
            if need_augmentation:
                img_list = self.image_augmentation(image, self.augmentation)
            else:
                img_list = [image]
            img_list = np.array(img_list, dtype=self.dtype)
            if self.mean is not None:
                img_list -= self.mean
            if self.std is not None:
                img_list /= self.std
            if self.transpose_axis:
                img_list = np.transpose(img_list, (0, 3, 1, 2))
            gal_images[di:di+aug_num, ...] = img_list
            gal_labels[di:di+aug_num] = pid_label[pid]
            di += aug_num
        print(' ... Done.')
        return prb_images, gal_images, prb_labels, gal_labels


class Market1501Reader(ImageReader):
    ih = 128
    iw = 64
    ich = 3
    train_person_num = 751
    test_person_num = 750
    def __init__(self, data_dir, re_size=None, save_dir=None, dtype=np.float32,
                 transpose_axis=True,
                 augmentation={'mirror':None,
                               'crop':None,
                               'crop_size':None,
                               'value_change':None,
                               'saturation_change':None}):
        super(Market1501Reader, self).__init__(data_dir, re_size, dtype, transpose_axis,
                                               augmentation)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
        self.aug_num = self.get_augmentation_num(augmentation)
        self.save_dir = save_dir
        
        self.train_bbox_dir = pathjoin(data_dir, 'bounding_box_train')
        self.test_bbox_dir = pathjoin(data_dir, 'bounding_box_test')
        self.query_dir = pathjoin(data_dir, 'query')
        
        self.num_class = self.train_person_num
        self.train_pid_label = {}
        self.train_label = 0
        self.test_pid_label = {}
        self.test_label = 0
    
    def read_images(self, part, need_augmentation=True,
                    include_distractors=True, need_shuffle=True):
        
        if part == 'train':
            files = glob.glob(pathjoin(self.train_bbox_dir, '*.jpg'))
        elif part == 'test':
            if not include_distractors:
                distracted_files = glob.glob(pathjoin(self.test_bbox_dir, '*.jpg'))
                files = []
                for f in distracted_files:
                    fname = f[f.rfind('/')+1:]
                    if fname[0] == '-' or fname[:4] == '0000':
                        continue
                    files.append(f)
            else:
                files = glob.glob(pathjoin(self.test_bbox_dir, '*.jpg'))
        elif part == 'query':
            files = glob.glob(pathjoin(self.query_dir, '*.jpg'))
        files.sort()
        
        n = len(files)
        if self.augmentation['crop_size'] is not None:
            sz = self.augmentation['crop_size'] + (self.ich,)
        elif self.re_size is not None:
            sz = self.re_size + (self.ich,)
        else:
            sz = (self.ih, self.iw, self.ich)
        if self.transpose_axis:
            sz = (sz[2], sz[0], sz[1])
        
        if need_augmentation:
            aug_num = self.aug_num
        else:
            aug_num = 1
        
        images = np.zeros((n*aug_num,) + sz, dtype=self.dtype)
        labels = np.zeros((n*aug_num), dtype=int)
        person_cam_index = np.zeros((n*aug_num, 2), dtype=int)
        
        if part == 'test' or part == 'query':
            pid_label = self.test_pid_label
            label = self.test_label
        elif part == 'train':
            pid_label = self.train_pid_label
            label = self.train_label
        di = 0
        for fi, f in enumerate(files):
            print('\rMarket-1501 Reader: Reading %s images %04d / %04d'%(part, fi+1, n), end='')
            fname = f[f.rfind('/')+1:]
            pid = fname[:4]
            if pid not in pid_label.keys():
                if pid[0] == '-' or pid == '0000':
                    pid_label[pid] = -1
                else:
                    pid_label[pid] = label
                    label += 1
            
            cam = int(fname[6])
            
            image = skio.imread(f)
            if not self.dtype == np.uint8:
                image = skimage.img_as_float( image )
            if self.re_size is not None:
                image = sktr.resize(image, self.re_size, mode='reflect')
            if need_augmentation:
                img_list = self.image_augmentation(image, self.augmentation)
            else:
                img_list = [image]
            img_list = np.array(img_list, self.dtype)
            if not self.dtype == np.uint8:
                img_list -= self.mean
                img_list /= self.std
            if self.transpose_axis:
                img_list = np.transpose(img_list, (0, 3, 1, 2))
            images[di:di+aug_num, ...] = img_list
            labels[di:di+aug_num] = pid_label[pid]
            person_cam_index[di:di+aug_num, 0] = pid_label[pid]
            person_cam_index[di:di+aug_num, 1] = cam
            di += aug_num
        
        if part == 'test' or part == 'query':
            self.test_pid_label = pid_label
            self.test_label = label
        elif part == 'train':
            self.train_pid_label = pid_label
            self.train_label = label
        
        if need_shuffle:
            ### shuffle X, Y and person-cam-index
            idxes = np.arange(images.shape[0], dtype=int)
            np.random.shuffle(idxes)
            images = images[idxes, ...]
            labels = labels[idxes]
            person_cam_index = person_cam_index[idxes, ...]
        print(' ... Done.')
        
        return images, labels, person_cam_index

class DukeMTMCReader(ImageReader):
    ich = 3
    train_person_num = 702
    test_person_num = 702
    def __init__(self, data_dir, re_size, save_dir=None, dtype=np.float32,
                 transpose_axis=True,
                 augmentation={'mirror':None,
                               'crop':None,
                               'crop_size':None,
                               'value_change':None,
                               'saturation_change':None}):
        super(DukeMTMCReader, self).__init__(data_dir, re_size, dtype, transpose_axis,
                                               augmentation)
    
        self.aug_num = self.get_augmentation_num(augmentation)
        self.save_dir = save_dir
        
        self.train_bbox_dir = pathjoin(data_dir, 'bounding_box_train')
        self.test_bbox_dir = pathjoin(data_dir, 'bounding_box_test')
        self.query_dir = pathjoin(data_dir, 'query')
        
        self.num_class = self.train_person_num
        self.train_pid_label = {}
        self.train_label = 0
        self.test_pid_label = {}
        self.test_label = 0
    
    def get_data_info(self):
        ''' 
            source: 'labeled' or 'detected'.
            trial: int type.
        '''
        if self.save_dir is not None\
        and os.path.exists(pathjoin(self.save_dir, 'data_info.pkl')):
            print('DukeMTMC-reID Reader: Loading data info.')
            with open(pathjoin(self.save_dir, 'data_info.pkl'), 'rb') as f:
                info = pickle.load(f)
            self.mean = info['mean']
            self.std = info['std']
            self.min_h = info['min_h']
            self.min_w = info['min_w']
            self.max_h = info['max_h']
            self.max_w = info['max_w']
        else:
            print('DukeMTMC-reID Reader *** Warning ***: New data infomation will recover the old one.')
            self.mean = np.zeros((3), dtype=self.dtype)
            self.std = np.zeros((3), dtype=self.dtype)
            self.min_h = 1000
            self.min_w = 1000
            self.max_h = 0
            self.max_w = 0
            
            files = glob.glob(pathjoin(self.train_bbox_dir, '*.jpg'))
            files.sort()
            fn = len(files)
            
            tr_count = 0
            for fi, f in enumerate(files):
                print('\rDukeMTMC-reID Reader: read info from image %05d / %05d'%(fi+1, fn), end='')
                img = skimage.img_as_float( skio.imread(f) )
                image = sktr.resize(img, self.re_size, mode='reflect')
                h, w, _ = img.shape
                if h < self.min_h:
                    self.min_h = h
                if h > self.max_h:
                    self.max_h = h
                if w < self.min_w:
                    self.min_w = w
                if w > self.max_w:
                    self.max_w = w
                
                self.mean[0] += np.mean(image[..., 0])
                self.mean[1] += np.mean(image[..., 1])
                self.mean[2] += np.mean(image[..., 2])
                tr_count += 1
            self.mean /= tr_count
            print(' ... Done.')
            
            if self.re_size is not None\
            and (self.min_h < self.re_size[0] or self.min_w < self.re_size):
                print('\rDukeMTMC-reID Reader *** Warning ***: Minimal image size smaller than reshape size.\n')
            
            for fi, f in enumerate(files):
                print('\rDukeMTMC-reID Reader: computing std %05d / %05d'%(fi+1, fn), end='')
                
                img = skimage.img_as_float( skio.imread(f) )
                img = sktr.resize(img, self.re_size, mode='reflect')
                
                self.std[0] += np.mean( (img[..., 0] - self.mean[0])**2 )
                self.std[1] += np.mean( (img[..., 1] - self.mean[1])**2 )
                self.std[2] += np.mean( (img[..., 2] - self.mean[2])**2 )
            self.std /= tr_count
            self.std = np.sqrt(self.std)
            print(' ... Done.')
            
            if self.save_dir is not None:
                with open(pathjoin(self.save_dir, 'data_info.pkl'), 'wb') as f:
                    pickle.dump({'mean':        self.mean,
                                 'std':         self.std,
                                 'min_h':       self.min_h,
                                 'min_w':       self.min_w,
                                 'max_h':       self.max_h,
                                 'max_w':       self.max_w}, f)
    
    def get_minimal_size(self):
        return self.min_h, self.min_w
    
    def set_reshape_size(self, re_size):
        self.re_size = re_size
    
    def set_minimal_size_as_reshape_size(self):
        self.re_size = (self.min_h, self.min_w)
    
    def read_images(self, part, need_augmentation=True, need_shuffle=True):
        
        if part == 'train':
            files = glob.glob(pathjoin(self.train_bbox_dir, '*.jpg'))
        elif part == 'test':
            files = glob.glob(pathjoin(self.test_bbox_dir, '*.jpg'))
        elif part == 'query':
            files = glob.glob(pathjoin(self.query_dir, '*.jpg'))
        files.sort()
        
        n = len(files)
        if self.augmentation['crop_size'] is not None:
            sz = self.augmentation['crop_size'] + (self.ich,)
        elif self.re_size is not None:
            sz = self.re_size + (self.ich,)
        else:
            sz = (self.ih, self.iw, self.ich)
        if self.transpose_axis:
            sz = (sz[2], sz[0], sz[1])
        
        if need_augmentation:
            aug_num = self.aug_num
        else:
            aug_num = 1
        
        images = np.zeros((n*aug_num,) + sz, dtype=self.dtype)
        labels = np.zeros((n*aug_num), dtype=int)
        person_cam_index = np.zeros((n*aug_num, 2), dtype=int)
        
        if part == 'test' or part == 'query':
            pid_label = self.test_pid_label
            label = self.test_label
        elif part == 'train':
            pid_label = self.train_pid_label
            label = self.train_label
        di = 0
        for fi, f in enumerate(files):
            print('\rDukeMTMC-reID Reader: Reading %s images %04d / %04d'%(part, fi+1, n), end='')
            fname = f[f.rfind('/')+1:]
            pid = fname[:4]
            if pid not in pid_label.keys():
                pid_label[pid] = label
                label += 1
            
            cam = int(fname[6])
            
            image = skio.imread(f)
            if not self.dtype == np.uint8:
                image = skimage.img_as_float( image )
#            if self.re_size is not None:
            image = sktr.resize(image, self.re_size, mode='reflect')
            
            if need_augmentation:
                img_list = self.image_augmentation(image, self.augmentation)
            else:
                img_list = [image]
            img_list = np.array(img_list, self.dtype)
            if not self.dtype == np.uint8:
                img_list -= self.mean
                img_list /= self.std
            if self.transpose_axis:
                img_list = np.transpose(img_list, (0, 3, 1, 2))
            images[di:di+aug_num, ...] = img_list
            labels[di:di+aug_num] = pid_label[pid]
            person_cam_index[di:di+aug_num, 0] = pid_label[pid]
            person_cam_index[di:di+aug_num, 1] = cam
            di += aug_num
        
        if part == 'test' or part == 'query':
            self.test_pid_label = pid_label
            self.test_label = label
        elif part == 'train':
            self.train_pid_label = pid_label
            self.train_label = label
        
        if need_shuffle:
            ### shuffle X, Y and person-cam-index
            idxes = np.arange(images.shape[0], dtype=int)
            np.random.shuffle(idxes)
            images = images[idxes, ...]
            labels = labels[idxes]
            person_cam_index = person_cam_index[idxes, ...]
        print(' ... Done.')
        
        return images, labels, person_cam_index


def concat_images(prbX, galX, prbY, galY):
    
    X = np.concatenate((prbX, galX), axis=0)
    Y = np.concatenate((prbY, galY), axis=0)
    ### index for pair batch
    person_cam_index = np.zeros((Y.shape[0], 2), dtype=int)
    person_cam_index[:, 0] = Y
    person_cam_index[:prbY.shape[0], 1] = 1
    person_cam_index[prbY.shape[0]:, 1] = 2
    ### shuffle X and Y and person-cam-index
    idxes = np.arange(X.shape[0], dtype=int)
    np.random.shuffle(idxes)
    X = X[idxes, ...]
    Y = Y[idxes]
    person_cam_index = person_cam_index[idxes, ...]
    
    return X, Y, person_cam_index


def grab_batch(X, Y, bat_start, batch_size):
    X_n = X.shape[0]
    bat_end = min(bat_start+batch_size, X_n)
    if bat_end - bat_start == batch_size:
        batch = X[bat_start:bat_end, ...]
        label = Y[bat_start:bat_end]
    else:
        batch = np.zeros((batch_size,) + X.shape[1:])
        label = np.zeros((batch_size,), dtype=int)
        part1_n = X_n - bat_start
        batch[:part1_n, ...] = X[bat_start:, ...]
        label[:part1_n] = Y[bat_start:]
        bat_end = batch_size - part1_n
        batch[part1_n:, ...] = X[:bat_end, ...]
        label[part1_n:] = Y[:bat_end]
    if bat_end == X_n:
        bat_start = 0
    else:
        bat_start = bat_end
    return batch, label, bat_start

def grab_pair_batch(X, Y, person_cam_index, num_class, batch_size):
    prb = np.zeros((batch_size,) + X.shape[1:])
    gal = np.zeros((batch_size,) + X.shape[1:])
    label = np.zeros((batch_size))

    pos_n = int(batch_size / 2)
    neg_n = batch_size - pos_n

    bat_i = 0
    for pos_i in range(pos_n):
        pid = np.random.randint(0, num_class)
        pid_idxes = np.where(person_cam_index[:,0] == pid)[0]
        cam1_idxes = np.where(person_cam_index[:,1] == 1)[0]
        cam2_idxes = np.where(person_cam_index[:,1] == 2)[0]
        prb_idxes = np.intersect1d(pid_idxes, cam1_idxes)
        gal_idxes = np.intersect1d(pid_idxes, cam2_idxes)
        prb_idx = prb_idxes[np.random.randint(0, len(prb_idxes))]
        gal_idx = gal_idxes[np.random.randint(0, len(gal_idxes))]
        prb[bat_i, ...] = X[prb_idx, ...]
        gal[bat_i, ...] = X[gal_idx, ...]
        label[bat_i] = 1
        bat_i += 1

    for neg_i in range(neg_n):
        ppid = np.random.randint(0, num_class)
        gpid = np.random.randint(0, num_class)
        while ppid == gpid:
            gpid = np.random.randint(0, num_class)
        ppid_idxes = np.where(person_cam_index[:,0] == ppid)[0]
        gpid_idxes = np.where(person_cam_index[:,0] == gpid)[0]
        cam1_idxes = np.where(person_cam_index[:,1] == 1)[0]
        cam2_idxes = np.where(person_cam_index[:,1] == 2)[0]
        prb_idxes = np.intersect1d(ppid_idxes, cam1_idxes)
        gal_idxes = np.intersect1d(gpid_idxes, cam2_idxes)
        prb_idx = prb_idxes[np.random.randint(0, len(prb_idxes))]
        gal_idx = gal_idxes[np.random.randint(0, len(gal_idxes))]
        prb[bat_i, ...] = X[prb_idx, ...]
        gal[bat_i, ...] = X[gal_idx, ...]
        label[bat_i] = 0
        bat_i += 1

    return prb, gal, label

def grab_triplet_batch(X, Y, person_cam_index, num_class, batch_size):
    anchor = np.zeros((batch_size,) + X.shape[1:])
    positive = np.zeros((batch_size,) + X.shape[1:])
    negative = np.zeros((batch_size,) + X.shape[1:])
    
    unique_cams = np.unique(person_cam_index[:, 1])
    cam_n = len(unique_cams)

    anchor_pids = []
    for bi in range(batch_size):
        ppid = np.random.randint(0, num_class)
        ''' Make sure different persons in a batch. '''
        if ppid in anchor_pids:
            ppid = np.random.randint(0, num_class)
        else:
            anchor_pids.append(ppid)

        gpid = np.random.randint(0, num_class)
        while ppid == gpid:
            gpid = np.random.randint(0, num_class)

        ppid_idxes = np.where(person_cam_index[:,0] == ppid)[0]
        gpid_idxes = np.where(person_cam_index[:,0] == gpid)[0]
        
        if cam_n == 2:
            anchor_cam_idxes = np.where(person_cam_index[:,1] == 1)[0]
            positive_cam_idxes = np.where(person_cam_index[:,1] == 2)[0]
            negative_cam_idxes = positive_cam_idxes
        else:
            p_cams = np.unique(person_cam_index[ppid_idxes,1])
            p_cam_n = len(p_cams)
            g_cams = np.unique(person_cam_index[gpid_idxes,1])
            g_cam_n = len(g_cams)
            
            anchor_cam = p_cams[np.random.randint(0, p_cam_n)]
            positive_cam = p_cams[np.random.randint(0, p_cam_n)]
            while anchor_cam == positive_cam:
                positive_cam = p_cams[np.random.randint(0, p_cam_n)]
            negative_cam = g_cams[np.random.randint(0, g_cam_n)]
            while anchor_cam == negative_cam:
                negative_cam = g_cams[np.random.randint(0, g_cam_n)]
            
            anchor_cam_idxes = np.where(person_cam_index[:,1] == anchor_cam)[0]
            positive_cam_idxes = np.where(person_cam_index[:,1] == positive_cam)[0]
            negative_cam_idxes = np.where(person_cam_index[:,1] == negative_cam)[0]
        

        anchor_idxes = np.intersect1d(ppid_idxes, anchor_cam_idxes)
        positive_idxes = np.intersect1d(ppid_idxes, positive_cam_idxes)
        negative_idxes = np.intersect1d(gpid_idxes, negative_cam_idxes)

        anchor_idx = anchor_idxes[np.random.randint(0, len(anchor_idxes))]
        positive_idx = positive_idxes[np.random.randint(0, len(positive_idxes))]
        negative_idx = negative_idxes[np.random.randint(0, len(negative_idxes))]

        anchor[bi, ...] = X[anchor_idx, ...]
        positive[bi, ...] = X[positive_idx, ...]
        negative[bi, ...] = X[negative_idx, ...]

    return anchor, positive, negative
