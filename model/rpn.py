#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : rpn.py
# Purpose :
# Creation Date : 10-12-2017
# Last Modified : 2017年12月13日 星期三 10时31分30秒
# Created By : Jialin Zhao

import tensorflow as tf
import numpy as np

from config import cfg
from misc_util import warn


small_addon_for_BCE=1e-6

class MiddleAndRPN:
    def __init__(self, input, alpha=1.5, beta=1, sigma=3, training=True, name=''):
        # scale = [batchsize, 10, 400/200, 352/240, 128] should be the output of feature learning network
        self.input = input  
        self.training = training
        # groundtruth(target) - each anchor box, represent as △x, △y, △z, △l, △w, △h, rotation
        self.targets = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14]) 
        # => wip: add confidence(iou) here for yolo style
        # => pos_equal_one is actually conf_mask in yolo code
        # self.conf_target = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2]) 
        # postive anchors equal to one and others equal to zero(2 anchors in 1 position)
        self.pos_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
        self.pos_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])
        self.pos_equal_one_for_reg = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14])
        # negative anchors equal to one and others equal to zero
        self.neg_equal_one = tf.placeholder(tf.float32, [None, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2])
        self.neg_equal_one_sum = tf.placeholder(tf.float32, [None, 1, 1, 1])

        with tf.variable_scope('MiddleAndRPN_' + name):
            #convolutinal middle layers
            temp_conv = ConvMD(3, 128, 64, 3, (2, 1, 1), (1, 1, 1), self.input, name='conv1')
            temp_conv = ConvMD(3, 64, 64, 3, (1, 1, 1), (0, 1, 1), temp_conv, name='conv2')
            temp_conv = ConvMD(3, 64, 64, 3, (2, 1, 1), (1, 1, 1), temp_conv, name='conv3')
            temp_conv = tf.transpose(temp_conv, perm = [0, 2, 3, 4, 1])
            temp_conv = tf.reshape(temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])
            # => batch, 400, 352, 128

            #rpn
            #block1:
            temp_conv = ConvMD(2, 128, 128, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv4')
            # => batch, 400, 352, 128
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool1')
            # => batch, 200, 176, 128
            temp_conv = ConvMD(2, 128, 256, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv5')
            # => batch, 200, 176, 256
            temp_conv = ConvMD(2, 256, 128, 1, (1, 1), (0, 0), temp_conv, training=self.training, name='conv6')
            # => batch, 200, 176, 128
            temp_conv = ConvMD(2, 128, 256, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv7')
            # => batch, 200, 176, 256
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool2')
            # => batch, 100, 88, 256
            temp_conv = ConvMD(2, 256, 512, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv8')
            # => batch, 100, 88, 512
            temp_conv = ConvMD(2, 512, 128, 1, (1, 1), (0, 0), temp_conv, training=self.training, name='conv9')
            # => batch, 100, 88, 128
            route_1 = ConvMD(2, 128, 256, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv10')
            # => batch, 100, 88, 256

            temp_conv = ConvMD(2, 256, 128, 1, (1, 1), (0, 0), route_1, training=self.training, name='conv11')
            # => batch, 100, 88, 128
            temp_conv = ConvMD(2, 128, 256, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv12')
            # => batch, 100, 88, 256
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool3')
            # => batch, 50, 44, 256
            temp_conv = ConvMD(2, 256, 512, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv13')
            # => batch, 50, 44, 512
            temp_conv = ConvMD(2, 512, 256, 1, (1, 1), (0, 0), temp_conv, training=self.training, name='conv14')
            # => batch, 50, 44, 256
            route_2 = ConvMD(2, 256, 512, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv15')
            # => batch, 50, 44, 512

            temp_conv = ConvMD(2, 256, 64, 3, (1, 1), (1, 1), route_1, training=self.training, name='conv16')
            # warn("shape: {}".format(np.shape(temp_conv)))
            # => batch, 100, 88, 64
            temp_conv = Reorg(2, temp_conv, name = 'reorg1')
            # => batch, 50, 44, 256
            temp_conv = tf.concat([temp_conv, route_2], axis = -1, name = 'concat1')
            # => batch, 50, 44, 768
            temp_conv = ConvMD(2, 768, 128, 3, (1, 1), (1, 1), temp_conv, training=self.training, name='conv17')
            # => batch, 50, 44, 128
            p_map = ConvMD(2, 128, 2, 1, (1, 1), (0, 0), temp_conv, training=self.training, name='conv18')
            r_map = ConvMD(2, 128, 14, 1, (1, 1), (0, 0), temp_conv, training=self.training, activation = False, name='conv19')

            self.p_pos = tf.sigmoid(p_map)
            self.output_shape = [cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH]

            x_pos_0 = tf.expand_dims(tf.sigmoid(r_map[..., 0]), -1)
            y_pos_0 = tf.expand_dims(tf.sigmoid(r_map[..., 1]), -1)
            x_pos_1 = tf.expand_dims(tf.sigmoid(r_map[..., 7]), -1)
            y_pos_1 = tf.expand_dims(tf.sigmoid(r_map[..., 8]), -1)

            r_map = tf.concat([x_pos_0, y_pos_0, r_map[:,:,:,2:7], x_pos_1, y_pos_1, r_map[:,:,:,9:14]], axis=-1)         

            warn("rmap shape:{}".format(np.shape(r_map)))
        
            # TODO: sometime still get inf cls loss
            # wip: change to yolo style

            object_scale = 1.0
            non_object_scale = 1.0


            # self.cls_loss = object_scale * (self.pos_equal_one * tf.square(self.p_pos - self.conf_target)) / self.pos_equal_one_sum\
            #                 + non_object_scale * self.neg_equal_one * tf.square(self.p_pos - self.conf_target) / self.neg_equal_one_sum
            # self.cls_loss = tf.reduce_sum(self.cls_loss)

            self.cls_loss = alpha * (-self.pos_equal_one * tf.log(self.p_pos + small_addon_for_BCE)) / self.pos_equal_one_sum \
             + beta * (-self.neg_equal_one * tf.pow(self.p_pos, 2.0) * tf.log(1 - self.p_pos + small_addon_for_BCE)) / self.neg_equal_one_sum
            self.cls_loss = tf.reduce_sum(self.cls_loss)

            # alpha_tf = 0.25
            # gamma = 2
            # pred_pt = tf.where(tf.equal(self.pos_equal_one, 1.0), self.p_pos, 1.0 - self.p_pos)
            # alpha_t = tf.scalar_mul(alpha_tf, tf.ones_like(self.pos_equal_one, dtype=tf.float32))
            # alpha_t = tf.where(tf.equal(self.pos_equal_one, 1.0), alpha_t, 1.0 - alpha_t)

            # self.focal_loss = tf.reduce_sum(-alpha_t * tf.pow(1.0 - pred_pt, gamma) * tf.log(pred_pt + small_addon_for_BCE))

            self.reg_loss = smooth_l1(r_map * self.pos_equal_one_for_reg, self.targets * self.pos_equal_one_for_reg, sigma) / self.pos_equal_one_sum
            self.reg_loss = tf.reduce_sum(self.reg_loss)

            self.loss = tf.reduce_sum(20.0 * self.cls_loss + self.reg_loss)

            self.delta_output = r_map 
            self.prob_output = self.p_pos


def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  =  tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    
    return smooth_l1

def Reorg(stride, x, name='reorg'):
    with tf.variable_scope(name) as scope:
        # warn("shape: {}".format(np.shape(x)))
        batch, height, width, channel = np.shape(x)
        # => batch * 100 * 88 * 64
        # warn("{} {} {} {}".format(batch, height, width, channel))
        x = tf.reshape(x, [-1, height//stride, stride, width//stride, stride, channel])
        # warn("shape: {}".format(np.shape(x)))

        # => batch * 50 * 2 * 44 * 2 * 64
        x = tf.transpose(x, perm = [0, 1, 3, 2, 4, 5])
        # warn("shape: {}".format(np.shape(x)))

        # => batch * 50 * 44 * 2 * 2 * 64 
        x = tf.reshape(x, [-1, height//stride, width//stride, 4*64])
        # warn("shape: {}".format(np.shape(x)))

        # => batch * 50 * 44 * 256
        return x

        #     temp_conv = tf.reshape(temp_conv, [-1, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128])

        # B = x.data.size(0)
        # C = x.data.size(1)
        # H = x.data.size(2)
        # W = x.data.size(3)
        # assert(H % stride == 0)
        # assert(W % stride == 0)
        # ws = stride
        # hs = stride
        # x = x.view(B, C, H/hs, hs, W/ws, ws).transpose(3,4).contiguous()
        # x = x.view(B, C, H/hs*W/ws, hs*ws).transpose(2,3).contiguous()
        # x = x.view(B, C, hs*ws, H/hs, W/ws).transpose(1,2).contiguous()
        # x = x.view(B, hs*ws*C, H/hs, W/ws)
        # return x



def ConvMD(M, Cin, Cout, k, s, p, input, training=True, activation=True, name='conv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values = (0, 0))
    with tf.variable_scope(name) as scope:
        if(M == 2):
            paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv2d(pad, Cout, k, strides = s, padding = "valid", reuse=tf.AUTO_REUSE, name=scope)
        if(M == 3):
            paddings = (np.array(temp_p)).repeat(2).reshape(5, 2)
            pad = tf.pad(input, paddings, "CONSTANT")
            temp_conv = tf.layers.conv3d(pad, Cout, k, strides = s, padding = "valid", reuse=tf.AUTO_REUSE, name=scope)
        temp_conv = tf.layers.batch_normalization(temp_conv, axis = -1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)

        if activation:
            return tf.nn.relu(temp_conv)
        else:
            return temp_conv

def Deconv2D(Cin, Cout, k, s, p, input, training=True, name='deconv'):
    temp_p = np.array(p)
    temp_p = np.lib.pad(temp_p, (1, 1), 'constant', constant_values = (0, 0))
    paddings = (np.array(temp_p)).repeat(2).reshape(4, 2)
    pad = tf.pad(input, paddings, "CONSTANT")
    with tf.variable_scope(name) as scope:
        temp_conv = tf.layers.conv2d_transpose(pad, Cout, k, strides = s, padding = "SAME", reuse=tf.AUTO_REUSE, name=scope)
        temp_conv = tf.layers.batch_normalization(temp_conv, axis = -1, fused=True, training=training, reuse=tf.AUTO_REUSE, name=scope)
        return tf.nn.relu(temp_conv)

if(__name__ == "__main__"):
    m = MiddleAndRPN(tf.placeholder(tf.float32, [None, 10, cfg.INPUT_HEIGHT, cfg.INPUT_WIDTH, 128]))
