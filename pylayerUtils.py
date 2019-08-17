#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import caffe
import numpy as np
import cv2
import random
import os
import math
import csv
from icdar import get_whole_data

class DataLayer(caffe.Layer):

    def setup(self, bottom, top):
        '''
        The Setup method is called once during the lifetime of the execution, when Caffe is instantiating all layers. This is where you will read parameters, instantiate fixed-size buffers.
        '''
        # data layer config
        params = eval(self.param_str)
        self.dataset = params['dataset']
        self.patch_size = 512
        self.batch_size = int(params['batch_size'])
        # self.mean = np.array(params['mean'])
        
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")
        if len(top) != 3:
            raise Exception("data top need three output")
            
        ## 数据集路径
        datasetDict = {
            'ic13': '/home1/surfzjy/data/ic13',
            'ic15': '/home1/surfzjy/data/ic15',
            'mlt': '/home1/surfzjy/data/ic17mlt',
            'hcp': '/home2/DATA/hcp_detection'
        }
        self.basedir = datasetDict[self.dataset] + '/train_images'
        self.fnLst = os.listdir(self.basedir)
        self.nb_img = len(self.fnLst)
        self.index = np.arange(0, self.nb_img)
        np.random.shuffle(self.index)
        self.idx = 0 

    def reshape(self, bottom, top):
        '''
        Use the reshape method for initialization/setup that depends on the bottom blob (layer input) size (for example top blob size and internal buffers). It is called before every forward.
        '''
        # reshape tops to fit (leading 1 is for batch dimension)
        self.data, self.score_map, self.geo_map = self.load()
        
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.score_map.shape)
        top[2].reshape(*self.geo_map.shape)

    def forward(self, bottom, top):
        '''
        The Forward method is called for each input batch and is where most of your logic will be.
        '''
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.score_map
        top[2].data[...] = self.geo_map

    def backward(self, top, propagate_down, bottom):
        '''
        The Backward method is called during the backward pass of the network. For example, in a convolution-like layer, this would be where you would calculate the gradients. This is optional (a layer can be forward-only).
        '''
        pass
        
    def load(self):
        # print('self.index = ', self.index)
        load_index = self.index[self.idx: min((self.idx+self.batch_size), self.nb_img)]
        # print('load_index = ', load_index)
        
        self.idx += self.batch_size
        self.idx = min(self.idx, self.nb_img - 1)
        
        return get_whole_data(input_size = self.patch_size,
                                    batch_size = self.batch_size,
                                    basedir = self.basedir,
                                    image_list = self.fnLst,
                                    load_index = load_index
                                    )
        # whole_data = get_whole_data(input_size = self.patch_size,
                                    # batch_size = self.batch_size,
                                    # basedir = self.basedir,
                                    # image_list = self.fnLst,
                                    # load_index = idx
                                    # )
        # input_images = (np.array(whole_data[0])).transpose(0,3,1,2)
        # input_score_maps = (np.array(whole_data[1])).transpose(0,3,1,2)
        # input_geo_maps = (np.array(whole_data[2])).transpose(0,3,1,2)
        # input_training_masks = (np.array(whole_data[4])).transpose(0,3,1,2)

        # return input_images, input_score_maps, input_geo_maps

class DiceCoefLossLayer(caffe.Layer):
    """
    self designed loss layer for segmentation. Class weighted, per pixel loss
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")
        self.batch_size = bottom[1].data.shape[0]

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        self.diff = np.zeros_like(bottom[0].data, dtype = np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[1].data
        self.sum = bottom[0].data.sum() + bottom[1].data.sum() + 1.
        self.dice = (2. * (bottom[0].data * bottom [1].data).sum() + 1.) / self.sum
        top[0].data[...] = 1.- self.dice

    def backward(self, top, propagate_down, bottom):
        if propagate_down[1]:
            raise Exception("label not diff")
        elif propagate_down[0]:
            bottom[0].diff[...] = (-2. * self.diff + 2. * bottom[0].data * self.dice) / self.sum
        else:
            raise Exception("no diff")

class RBoxLossLayer(caffe.Layer):
    """
    self designed loss layer for segmentation. Class weighted, per pixel loss
    bottom: "concat4"  # 1*5*128*128 (4+1)
    bottom: "geo_map"  # 1*5*128*128 (4+1)
    bottom: "score_map" # scoremap的ture_gt 1*1*128*128
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 3:
            raise Exception("Need three inputs to compute total Loss.")
        self.batch_size = bottom[1].data.shape[0]
        self.pixel_num = bottom[1].data.shape[2] * bottom[1].data.shape[3]
        self.ratio = 5.

    def reshape(self, bottom, top):
        # check input dimensions match
        # print(bottom[0].data.shape) # N*5*128*128 pred
        # print(bottom[1].data.shape) # N*5*128*128 geo_gt
        # print(bottom[2].data.shape) # N*1*128*128 score_gt
        if bottom[0].count != bottom[1].count:
            raise Exception("First Two Inputs must have the same dimension.")
        # self.geo_gt = np.zeros_like(bottom[1].data,dtype=np.float32)
        self.score_gt = np.zeros_like(bottom[2].data, dtype = np.float32)
        self.L_g = np.zeros_like(bottom[0].data[:,0,:,:], dtype = np.float32)
        self.top_grad1 = np.zeros_like(bottom[2].data, dtype = np.float32)
        self.top_grad2 = np.zeros_like(bottom[2].data, dtype = np.float32)
        self.L_theta_grad = np.zeros_like(bottom[2].data, dtype = np.float32)

        # loss output is scalar
        top[0].reshape(1) # 1,

    def forward(self, bottom, top):
        self.score_gt[...] = bottom[2].data  

        self.d1_pred, self.d2_pred, self.d3_pred, self.d4_pred, self.theta_pred = np.array_split(bottom[0].data, indices_or_sections = 5, axis = 1)
        self.d1_gt, self.d2_gt, self.d3_gt, self.d4_gt, self.theta_gt = np.array_split(bottom[1].data, indices_or_sections = 5, axis = 1)
                
        area_gt = (self.d1_gt + self.d3_gt) * (self.d2_gt + self.d4_gt)
        area_pred = (self.d1_pred + self.d3_pred) * (self.d2_pred + self.d4_pred)
        self.w_union = np.minimum(self.d2_gt, self.d2_pred) + np.minimum(self.d4_gt, self.d4_pred)
        self.h_union = np.minimum(self.d1_gt, self.d1_pred) + np.minimum(self.d3_gt, self.d3_pred)
        self.area_intersect = self.w_union * self.h_union
        self.area_union = area_gt + area_pred - self.area_intersect      
        
        L_theta = 1. - np.cos(self.theta_pred - self.theta_gt) 
        L_AABB = -np.log((self.area_intersect + 1.) / (self.area_union + 1.))
        self.L_g = L_AABB + self.ratio * L_theta 
        top[0].data[...] = np.mean(self.L_g * self.score_gt)
        
    def backward(self, top, propagate_down, bottom):

        ai_grad1 = self.w_union * (1. * (self.d1_pred <= self.d1_gt)) 
        self.top_grad1 = self.score_gt / self.pixel_num / self.batch_size * ((self.d2_pred + self.d4_pred - ai_grad1) / (self.area_union + 1.) - ai_grad1 / (self.area_intersect + 1.))

        ai_grad2 = self.h_union * (1. * (self.d2_pred <= self.d2_gt))
        self.top_grad2 = self.score_gt / self.pixel_num / self.batch_size * ((self.d1_pred + self.d3_pred - ai_grad2) / (self.area_union + 1.) - ai_grad2 / (self.area_intersect + 1.))
        
        ai_grad3 = self.w_union * (1. * (self.d3_pred <= self.d3_gt)) 
        self.top_grad3 = self.score_gt / self.pixel_num / self.batch_size * ((self.d2_pred + self.d4_pred - ai_grad3) / (self.area_union + 1.) - ai_grad3 / (self.area_intersect + 1.))
        
        ai_grad4 = self.h_union * (1. * (self.d4_pred <= self.d4_gt))
        self.top_grad4 = self.score_gt / self.pixel_num / self.batch_size * ((self.d1_pred + self.d3_pred - ai_grad4) / (self.area_union + 1.) - ai_grad4 / (self.area_intersect + 1.))
        
        self.L_theta_grad = self.ratio * self.score_gt / self.pixel_num / self.batch_size * np.sin(self.theta_pred - self.theta_gt)
                
        bottom[0].diff[...] = np.concatenate((self.top_grad1, self.top_grad2, self.top_grad3, self.top_grad4, self.L_theta_grad), axis=1)
        bottom[1].diff[...] = 0
        bottom[2].diff[...] = 0     
