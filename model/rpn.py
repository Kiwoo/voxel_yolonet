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
from utils.utils import *


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
            warn("rmap shape:{}".format(np.shape(r_map)))

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
            self.corner_loss = tf.cond(tf.equal(tf.shape(self.targets * self.pos_equal_one_for_reg)[0], 0), lambda: return_zero(), \
                lambda: cal_volume_loss(r_map * self.pos_equal_one_for_reg, self.targets * self.pos_equal_one_for_reg, self.pos_equal_one))
            # self.corner_loss = tf.reduce_sum(self.corner_loss)
            self.loss = tf.reduce_sum(1.0 * self.cls_loss) + tf.reduce_sum(10.0*self.corner_loss)

            self.delta_output = r_map 
            self.prob_output = self.p_pos

def return_zero():
    return 0.0

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)

    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    
    return smooth_l1

def tf_cal_anchors():
    anchors = tf.constant(cal_anchors(), dtype=tf.float32)
    return anchors

def tf_delta_to_boxes3d(deltas, anchors, flipped=False):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    anchors_reshaped = tf.reshape(anchors, [-1, 7])
    deltas = tf.reshape(deltas, [-1, cfg.FEATURE_WIDTH*cfg.FEATURE_HEIGHT*2, 7])
    anchors_d = tf.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    boxes3d = tf.zeros_like(deltas)

    x = deltas[..., 0] * cfg.FEATURE_WIDTH_ACTUAL + anchors_reshaped[..., 0]
    y = deltas[..., 1] * cfg.FEATURE_HEIGHT_ACTUAL + anchors_reshaped[..., 1]
    z = deltas[..., 2] * cfg.ANCHOR_H + anchors_reshaped[..., 2]
    h = tf.exp(deltas[..., 3]) * anchors_reshaped[..., 3]
    w = tf.exp(deltas[..., 4]) * anchors_reshaped[..., 4]
    l = tf.exp(deltas[..., 5]) * anchors_reshaped[..., 5]
    if flipped == True:
        flipped_angle = tf.ones_like(anchors_reshaped[..., 6]) * np.pi
        r = deltas[..., 6] + anchors_reshaped[..., 6] + flipped_angle
    else:        
        r = deltas[..., 6] + anchors_reshaped[..., 6]
    boxes3d = tf.stack([x, y, z, h, w, l, r], axis=-1)

    return boxes3d

def tf_corner_to_standup(boxes_corner_a, boxes_corner_b):
    # boxes_corner_a: (N,8,3)

    boxes_a_x = boxes_corner_a[:,0:4,0] # N,4
    boxes_a_y = boxes_corner_a[:,0:4,1]
    boxes_b_x = boxes_corner_b[:,0:4,0]
    boxes_b_y = boxes_corner_b[:,0:4,1]

    warn("boxes_a_x: {}".format(np.shape(boxes_a_x)))

    x_axis = tf.concat([boxes_a_x, boxes_b_x], axis=1) # N,8
    y_axis = tf.concat([boxes_a_y, boxes_b_y], axis=1) # N,8

    warn("x_axis: {}".format(np.shape(x_axis)))


    min_x = tf.reduce_min(x_axis, axis=1)    # N
    min_y = tf.reduce_min(y_axis, axis=1)
    max_x = tf.reduce_max(x_axis, axis=1)
    max_y = tf.reduce_max(y_axis, axis=1)

    warn("min_x: {}".format(np.shape(min_x)))

    translation_x = tf.tile(tf.expand_dims(min_x, -1), [1,4])
    translation_y = tf.tile(tf.expand_dims(min_y, -1), [1,4])
    warn("translation_x: {}".format(np.shape(translation_x)))

    boxes_a_x = boxes_a_x - translation_x
    boxes_b_x = boxes_b_x - translation_x
    boxes_a_y = boxes_a_y - translation_y
    boxes_b_y = boxes_b_y - translation_y

    len_x = max_x - min_x
    len_y = max_y - min_y # N,1

    warn("len_x: {}".format(np.shape(len_x)))

    len_square = tf.maximum(len_x, len_y)    # N,1
    warn("len_square 1: {}".format(np.shape(len_square)))
    len_square = tf.tile(tf.expand_dims(len_square, -1), [1,4]) # N,4
    warn("len_square 2: {}".format(np.shape(len_square)))
   
    boxes_a_x = tf.divide(boxes_a_x, len_square) # N,4
    boxes_b_x = tf.divide(boxes_b_x, len_square)
    boxes_a_y = tf.divide(boxes_a_y, len_square)
    boxes_b_y = tf.divide(boxes_b_y, len_square)

    boxes_a = tf.stack([boxes_a_x, boxes_a_y], axis=-1) # N,4,2
    boxes_b = tf.stack([boxes_b_x, boxes_b_y], axis=-1) # N,4,2
    warn("boxes_a_x: {}".format(np.shape(boxes_a)))

    # standup_boxes = tf.stack([min_x, min_y, max_x, max_y], axis=1) # N,4

    return boxes_a, boxes_b


    # N = boxes_corner.shape[0]
    # standup_boxes2d = np.zeros((N, 4))
    # standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    # standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    # standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    # standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    # return standup_boxes2d

def tf_random_gen_points():
    # boxes: (N, 4) min_x, min_y, max_x, max_y

    num_points = 25
    x = tf.lin_space(0.0, 1.0, num_points)
    y = tf.lin_space(0.0, 1.0, num_points)
    xx,yy = tf.meshgrid(x,y)
    xx = tf.reshape(xx, [-1, 1])
    yy = tf.reshape(yy, [-1, 1])
    grid_points = tf.concat([xx,yy], axis=1)

    return grid_points

def tf_rot_points(points, rot_angle):

    # points: N,NP,2 (NP for number of points, 2 for x, y)
    # rot_angle: N,
    cos_r = tf.cos(rot_angle)
    sin_r = tf.sin(rot_angle)

    col_1 = tf.stack([cos_r, sin_r], axis=1) # N,2
    col_2 = tf.stack([-sin_r, cos_r], axis=1)

    # this is for making Nx2x2 rotation matrix along z axis
    # [[ cos, -sin],
    #  [ sin, cos]]

    rotMat = tf.stack([col_1, col_2], axis=-1) # N,2,2
    rot_points = tf.matmul(points, rotMat)
    return rot_points # N, NP, 2

def tf_points_in_boxes(boxes, points):
    # boxes: N,4,2
    # points: N,NP,2
    # return: N,NP boolean or 0,1 to indicate whether the each point exists inside boxes or not.

    boxes_x = boxes[:,:,0] # N,4
    boxes_y = boxes[:,:,1] # N,4

    min_x = tf.reduce_min(boxes_x, axis=1)    # N,
    max_x = tf.reduce_max(boxes_x, axis=1)
    min_y = tf.reduce_min(boxes_y, axis=1)
    max_y = tf.reduce_max(boxes_y, axis=1)   


    points_x = points[:,:,0] # N,NP
    points_y = points[:,:,1]

    points_inside = tf.zeros_like(points_x) # N,NP



    min_x = tf.expand_dims(min_x, -1)
    max_x = tf.expand_dims(max_x, -1)
    min_y = tf.expand_dims(min_y, -1)
    max_y = tf.expand_dims(max_y, -1)   

    minx = tf.greater_equal(points_x, min_x) # N,NP
    maxx = tf.less_equal(points_x, max_x)
    miny = tf.greater_equal(points_y, min_y)
    maxy = tf.less_equal(points_y, max_y)

    x_cond = tf.logical_and(minx, maxx)
    y_cond = tf.logical_and(miny, maxy)
    points_in_boxes = tf.logical_and(x_cond, y_cond)

    # inside = tf.ones_like(points_x)
    # outside = tf.zeros_like(points_x)

    # points_in_boxes = tf.where(rec_cond, inside, outside)
    warn("points_in_boxes: {}".format(np.shape(points_in_boxes)))

    return points_in_boxes

def tf_calculate_rotation_iou(boxes_corner_a, boxes_center_a, boxes_corner_b, boxes_center_b):
    # boxes_corner_a : predicted corner boxes => N,8,3
    # boxes_corner_b : ground truth boxes     => N,8,3
    # boxes_center_a : predicted center boxes => N,7
    # boxes_center_b : ground truth boxes => N,7

    # (1) get max boundaries

    # input: boxes_corner_a[N, 8, 3], boxes_corner_b[N, 8, 2] only x and y
    # output: max_boundaries: 2 * (N,4,2) => 2 boxes, 4 points, x and y 

    boxes_standup_a, boxes_standup_b = tf_corner_to_standup(boxes_corner_a, boxes_corner_b)


    # (2) distribute points onto maximum boundaries
    # number of points to be distributed: 20 x 20 => 400
    # output: num_of_points: (NP, 2): NP: number of points, 2: x, y

    grid_points = tf_random_gen_points()
    grid_points = tf.expand_dims(grid_points, 0)

    N = tf.shape(boxes_standup_a)[0]
    warn("grid_points: {}".format(np.shape(grid_points)))

    grid_points = tf.tile(grid_points, [N,1,1]) # N,400,2
    warn("grid_points: {}".format(np.shape(grid_points)))

    # (3) first rotation

    # input: num_of_points:(N,NP,2), boxes_corner_a:(N,4,2), and boxes_corner_b:(N,4,2), rotation: N

    angle_a = boxes_center_a[:,6] # (N,)
    rot_a = np.pi/2 - angle_a # (N,)
    rot_boxes_standup_a = tf_rot_points(boxes_standup_a, rot_a)
    rot_points_grid = tf_rot_points(grid_points, rot_a)
    # (3-1) select points
    points_in_boxes_a = tf_points_in_boxes(rot_boxes_standup_a, rot_points_grid) # N,NP
    warn("boxes_center_a: {}".format(np.shape(boxes_center_a)))
    warn("angle_a: {}".format(np.shape(angle_a)))

    warn("rot_a: {}".format(np.shape(rot_a)))

    # (4) second rotation
    angle_b = boxes_center_b[:,6] # (N,)
    rot_b = np.pi/2 - angle_b # (N,)
    rot_boxes_standup_b = tf_rot_points(boxes_standup_b, rot_b)
    rot_points_grid = tf_rot_points(grid_points, rot_b)
    # (4-1) select points
    points_in_boxes_b = tf_points_in_boxes(rot_boxes_standup_b, rot_points_grid) # N,NP
    warn("boxes_center_b: {}".format(np.shape(boxes_center_b)))
    warn("angle_b: {}".format(np.shape(angle_b)))

    warn("rot_b: {}".format(np.shape(rot_b)))    # (4-1) select points

    points_in_intersection = tf.logical_and(points_in_boxes_a, points_in_boxes_b)

    warn("points_in_intersection: {}".format(np.shape(points_in_intersection)))

    # (5) calculate ratio between number of points in union and number of points in intersection


    inside = tf.ones_like(points_in_boxes_a, tf.float32)
    outside = tf.zeros_like(points_in_boxes_a, tf.float32)
    num_points_in_boxes_a = tf.where(points_in_boxes_a, inside, outside)
    warn("num_points_in_boxes_a 1: {}".format(np.shape(num_points_in_boxes_a)))
    num_points_in_boxes_a = tf.reduce_sum(num_points_in_boxes_a, axis=1) # N,
    warn("num_points_in_boxes_a 2: {}".format(np.shape(num_points_in_boxes_a)))
    num_points_in_boxes_b = tf.where(points_in_boxes_b, inside, outside)
    num_points_in_boxes_b = tf.reduce_sum(num_points_in_boxes_b, axis=1)
    num_points_in_intersection = tf.where(points_in_intersection, inside, outside)
    warn("num_points_in_intersection 1: {}".format(np.shape(num_points_in_intersection))) # N,400
    num_points_in_intersection = tf.reduce_sum(num_points_in_intersection, axis=1) # N,

    num_points_in_union = num_points_in_boxes_a + num_points_in_boxes_b - num_points_in_intersection

    iou = tf.divide(num_points_in_intersection, num_points_in_union) 
    warn("iou: {}".format(np.shape(iou)))

    return iou





    # (6) return iou




def tf_center_to_corner_box3d(boxes_center, coordinate='lidar'):
    # (N, 7) -> (N, 8, 3)

    box = boxes_center
    translation = box[:, 0:3]
    size = box[:, 3:6]
    yaw = box[:, -1]
    h, w, l = size[:, 0], size[:, 1], size[:, 2]

    ml = -l/2
    pl = l/2
    mw = -w/2
    pw = w/2
    zh = tf.zeros_like(pw, tf.float32)
    ph = h

    p1 = tf.stack([ml, pw, zh], axis=1)
    p2 = tf.stack([ml, mw, zh], axis=1)
    p3 = tf.stack([pl, mw, zh], axis=1)
    p4 = tf.stack([pl, pw, zh], axis=1)
    p5 = tf.stack([ml, pw, ph], axis=1)
    p6 = tf.stack([ml, mw, ph], axis=1)
    p7 = tf.stack([pl, mw, ph], axis=1)
    p8 = tf.stack([pl, pw, ph], axis=1)

    trackletBox = tf.stack([p1, p2, p3, p4, p5, p6, p7, p8], axis=-1)
  
    cos_y = tf.cos(yaw)
    sin_y = tf.sin(yaw)
    zero_y = tf.zeros_like(yaw, tf.float32)
    ones_y = tf.ones_like(yaw, tf.float32)

    r1 = tf.stack([cos_y, sin_y, zero_y], axis=1)
    r2 = tf.stack([-sin_y, cos_y, zero_y], axis=1)
    r3 = tf.stack([zero_y, zero_y, ones_y], axis=1)

    rotMat = tf.stack([r1, r2, r3], axis=-1)

    translation = tf.expand_dims(translation, -1)
    translation = tf.tile(translation, [1,1,8])

    cornerPosInVelo = tf.matmul(rotMat, trackletBox) + translation
    cornerPosInVelo = tf.transpose(cornerPosInVelo, [0,2,1])

    return cornerPosInVelo

def cal_volume_loss(delta_a, delta_b, mask):
    loss = 0.0
    sigma = 3.0
    anchors = tf_cal_anchors()
    batch_boxes3d_a = tf_delta_to_boxes3d(delta_a, anchors) # prediction
    batch_boxes3d_b = tf_delta_to_boxes3d(delta_b, anchors) # ground truth
    batch_boxes3d_b_flipped = tf_delta_to_boxes3d(delta_b, anchors, True) # ground truth flipped

    mask = tf.reshape(mask, [-1, cfg.FEATURE_WIDTH*cfg.FEATURE_HEIGHT*2])#mask.reshape((batch_size, -1))

    ind = tf.equal(mask[:, :], 1.0)
    batch_boxes3d_a = tf.reshape(batch_boxes3d_a, [-1, 7])
    batch_boxes3d_b = tf.reshape(batch_boxes3d_b, [-1, 7])
    batch_boxes3d_b_flipped = tf.reshape(batch_boxes3d_b_flipped, [-1, 7])
    ind = tf.reshape(ind, [-1])

    center_boxes3d_a = tf.boolean_mask(batch_boxes3d_a, ind) # N, 7
    center_boxes3d_b = tf.boolean_mask(batch_boxes3d_b, ind)
    center_boxes3d_b_flipped = tf.boolean_mask(batch_boxes3d_b_flipped, ind)

    corner_boxes3d_a = tf_center_to_corner_box3d(center_boxes3d_a, coordinate='lidar') 
    corner_boxes3d_b = tf_center_to_corner_box3d(center_boxes3d_b, coordinate='lidar') 

    iou = tf_calculate_rotation_iou(corner_boxes3d_a, center_boxes3d_a, corner_boxes3d_b, center_boxes3d_b)


    corner_boxes3d_b_flipped = tf_center_to_corner_box3d(center_boxes3d_b_flipped, coordinate='lidar') 


    warn("smooth loss: {}".format(np.shape(loss)))
    loss = tf.minimum(tf.reduce_sum(smooth_l1(corner_boxes3d_a, corner_boxes3d_b, sigma), [1,2]), \
                    tf.reduce_sum(smooth_l1(corner_boxes3d_a, corner_boxes3d_b_flipped, sigma), [1,2]))

    warn("loss : {}".format(np.shape(loss)))
    warn("iou : {}".format(np.shape(iou)))
    a = tf.pow(1.0-iou, 2.0)
    warn("pow iou: {}".format(np.shape(a)))
    loss = tf.reduce_sum(loss * tf.pow(1.0-iou, 2.0))
    a = tf.reduce_sum(smooth_l1(corner_boxes3d_a, corner_boxes3d_b, sigma), [1,2])
    warn("one loss: {}".format(np.shape(a)))
    warn("loss : {}".format(np.shape(loss)))
    # * loss should be modified. I think this has some error in calculating 
    # * not reduce_sum and take minimum, but minimum first and do reduce_sum


    divider = tf.maximum(tf.shape(corner_boxes3d_a)[0], 1) # normalize by number of ground truth or prediction boxes
    # if we don't have this, then as we have more number of boxes, the loss will be larger.
    divider = tf.cast(divider, dtype=tf.float32)
    loss = tf.divide(loss, divider)
    return loss

def Reorg(stride, x, name='reorg'):
    with tf.variable_scope(name) as scope:
        batch, height, width, channel = np.shape(x)
        # => batch * 100 * 88 * 64
        x = tf.reshape(x, [-1, height//stride, stride, width//stride, stride, channel])
        # => batch * 50 * 2 * 44 * 2 * 64
        x = tf.transpose(x, perm = [0, 1, 3, 2, 4, 5])
        # => batch * 50 * 44 * 2 * 2 * 64 
        x = tf.reshape(x, [-1, height//stride, width//stride, 4*64])
        # => batch * 50 * 44 * 256
        return x


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
