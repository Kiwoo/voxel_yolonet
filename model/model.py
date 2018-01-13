#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import sys
import os
import tensorflow as tf

from config import cfg
from utils.utils import * 
from model.group_pointcloud import FeatureNet, FeatureNet_new
from model.rpn import MiddleAndRPN
from misc_util import warn
from utils.colorize import *


class RPN3D(object):

    def __init__(self,
            cls='Car',
            single_batch_size=2, # batch_size_per_gpu
            learning_rate=0.001,
            max_gradient_norm=5.0,
            alpha=1.5,
            beta=1,
            is_train=True,
            avail_gpus=['0']):
        # hyper parameters and status
        self.cls = cls 
        self.single_batch_size = single_batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(1, trainable=False)
        self.epoch = tf.Variable(0, trainable=False)
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)
        self.alpha = alpha 
        self.beta = beta 
        self.avail_gpus = avail_gpus

        lr = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.96)

        # build graph
        # input placeholders
        self.vox_feature = [] 
        self.vox_number = []
        self.vox_coordinate = [] 

        self.doubled_vox_feature = [] 
        self.doubled_vox_number = []
        self.doubled_vox_coordinate = [] 

        self.targets = [] 
        # => wip: add confidence(iou) here for yolo style
        # => pos_equal_one is actually conf_mask in yolo code
        # self.conf_target = []
        self.pos_equal_one = [] 
        self.pos_equal_one_sum = [] 
        self.pos_equal_one_for_reg = [] 
        self.neg_equal_one = [] 
        self.neg_equal_one_sum = []

        self.delta_output = []
        self.prob_output = []
        self.opt = tf.train.AdamOptimizer(lr)
        self.gradient_norm = []
        self.tower_grads = []

        self.total_loss = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
                    # must use name scope here since we do not want to create new variables
                    # graph
                    feature = FeatureNet(training=is_train, batch_size=self.single_batch_size, doubled = False, name = 'single_feat')
                    doubled_feature = FeatureNet(training=is_train, batch_size=self.single_batch_size, doubled = True, name = 'doubled_feat')
                    # warn("check size: {}".format(np.shape(doubled_feature.outputs)))
                    # concat_feature = tf.concat([feature.outputs, doubled_feature.outputs], axis = -1)
                    # warn("check size: {}".format(np.shape(concat_feature)))
                    concat_feat = tf.concat([feature.outputs, doubled_feature.outputs], axis = -1)
                    rpn = MiddleAndRPN(input=concat_feat, alpha=self.alpha, beta=self.beta, training=is_train)
                    tf.get_variable_scope().reuse_variables()
                    # input
                    self.vox_feature.append(feature.feature)
                    self.vox_number.append(feature.number)
                    self.vox_coordinate.append(feature.coordinate)

                    self.doubled_vox_feature.append(doubled_feature.feature)
                    self.doubled_vox_number.append(doubled_feature.number)
                    self.doubled_vox_coordinate.append(doubled_feature.coordinate)

                    self.targets.append(rpn.targets)
                    # self.conf_target.append(rpn.conf_target)
                    self.pos_equal_one.append(rpn.pos_equal_one)
                    self.pos_equal_one_sum.append(rpn.pos_equal_one_sum)
                    self.pos_equal_one_for_reg.append(rpn.pos_equal_one_for_reg)
                    self.neg_equal_one.append(rpn.neg_equal_one)
                    self.neg_equal_one_sum.append(rpn.neg_equal_one_sum)
                    # output
                    feature_output = feature.outputs
                    delta_output = rpn.delta_output 
                    prob_output = rpn.prob_output 
                    # loss and grad
                    self.loss = rpn.loss
                    self.reg_loss = rpn.reg_loss
                    self.cls_loss = rpn.cls_loss
                    # self.cls_loss1 = rpn.cls_loss1
                    self.params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

                    self.delta_output.append(delta_output)
                    self.prob_output.append(prob_output)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.rpn_output_shape = rpn.output_shape 

                    self.total_loss.append(self.loss)

        # loss and optimizer
        # self.xxxloss is only the loss for the lowest tower
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = average_gradients(self.tower_grads)
            self.update = self.opt.apply_gradients(zip(self.grads, self.params), global_step=self.global_step)
            self.gradient_norm = tf.group(*self.gradient_norm)

        self.delta_output = tf.concat(self.delta_output, axis=0)
        self.prob_output = tf.concat(self.prob_output, axis=0)

        self.anchors = cal_anchors()
        # for predict and image summary 
        self.rgb = tf.placeholder(tf.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])


        self.bv = tf.placeholder(tf.uint8, [
                                 None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3])
        self.bv_heatmap = tf.placeholder(tf.uint8, [
            None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3])
        self.boxes2d = tf.placeholder(tf.float32, [None, 4])
        self.boxes2d_scores = tf.placeholder(tf.float32, [None])


        # NMS(2D)
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.box2d_ind_after_nms = tf.image.non_max_suppression(self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
    
        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            tf.summary.scalar('train/reg_loss', self.reg_loss),
            tf.summary.scalar('train/cls_loss', self.cls_loss),
            [tf.summary.histogram(each.name, each) for each in self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss),
            tf.summary.scalar('validate/reg_loss', self.reg_loss),
            tf.summary.scalar('validate/cls_loss', self.cls_loss)
        ])

        # TODO: bird_view_summary and front_view_summary
        
        self.predict_summary = tf.summary.merge([
            tf.summary.image('predict/bird_view_lidar', self.bv),
            tf.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
            tf.summary.image('predict/front_view_rgb', self.rgb),
        ])


    def train_step(self, session, data, train=False, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        doubled_vox_feature = data[5]
        doubled_vox_number = data[6]
        doubled_vox_coordinate = data[7]
        print('train', tag)
        pos_equal_one, neg_equal_one, targets = cal_rpn_target(
            label, self.rpn_output_shape, self.anchors, cls=cfg.DETECT_OBJ, coordinate='lidar')
        pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
        pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1,2,3)).reshape(-1,1,1,1), a_min=1, a_max=None) 
        neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1,2,3)).reshape(-1,1,1,1), a_min=1, a_max=None)
        # warn("shape: {} {}".format(np.shape(vox_feature), np.shape(doubled_vox_feature))) 
        
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]

            input_feed[self.doubled_vox_feature[idx]] = doubled_vox_feature[idx]
            # warn("test: {}".format(vox_feature[idx][0][0:4]))
            input_feed[self.doubled_vox_number[idx]] = doubled_vox_number[idx]
            input_feed[self.doubled_vox_coordinate[idx]] = doubled_vox_coordinate[idx]
            # warn("===========================")
            # warn("train: {} {} {}".format(np.shape(vox_feature[idx]), np.shape(vox_number[idx]), np.shape(vox_coordinate[idx])))

            # warn("train: {} {} {}".format(np.shape(doubled_vox_feature[idx]), np.shape(doubled_vox_number[idx]), np.shape(doubled_vox_coordinate[idx])))

            input_feed[self.targets[idx]] = targets[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            # => wip: add confidence(iou) here for yolo style
            # => pos_equal_one is actually conf_mask in yolo code
            # input_feed[self.conf_target[idx]] = conf_target[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
            input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
        if train:
            output_feed = [self.loss, self.reg_loss, self.cls_loss, self.gradient_norm, self.update]
        else:
            output_feed = [self.loss, self.reg_loss, self.cls_loss]
        if summary:
            output_feed.append(self.train_summary)
        # TODO: multi-gpu support for test and predict step
        return session.run(output_feed, input_feed)


    # def validate_step(self, session, data, summary=False):
    #     # input:  
    #     #     (N) tag 
    #     #     (N, N') label
    #     #     vox_feature 
    #     #     vox_number 
    #     #     vox_coordinate
    #     tag = data[0]
    #     label = data[1]
    #     vox_feature = data[2]
    #     vox_number = data[3]
    #     vox_coordinate = data[4]

    #     print('valid', tag)
    #     pos_equal_one, neg_equal_one, targets = cal_rpn_target(label, self.rpn_output_shape, self.anchors)
    #     pos_equal_one_for_reg = np.concatenate([np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1)
    #     pos_equal_one_sum = np.clip(np.sum(pos_equal_one, axis=(1,2,3)).reshape(-1,1,1,1), a_min=1, a_max=None) 
    #     neg_equal_one_sum = np.clip(np.sum(neg_equal_one, axis=(1,2,3)).reshape(-1,1,1,1), a_min=1, a_max=None)
        
    #     input_feed = {}
    #     for idx in range(len(self.avail_gpus)):
    #         input_feed[self.vox_feature[idx]] = vox_feature[idx]
    #         input_feed[self.vox_number[idx]] = vox_number[idx]
    #         input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
    #         input_feed[self.targets[idx]] = targets[idx*self.single_batch_size:(idx+1)*self.single_batch_size]

    #         input_feed[self.pos_equal_one[idx]] = pos_equal_one[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
    #         input_feed[self.pos_equal_one_sum[idx]] = pos_equal_one_sum[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
    #         input_feed[self.pos_equal_one_for_reg[idx]] = pos_equal_one_for_reg[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
    #         input_feed[self.neg_equal_one[idx]] = neg_equal_one[idx*self.single_batch_size:(idx+1)*self.single_batch_size]
    #         input_feed[self.neg_equal_one_sum[idx]] = neg_equal_one_sum[idx*self.single_batch_size:(idx+1)*self.single_batch_size]

    #     output_feed = [self.loss, self.reg_loss, self.cls_loss]
    #     if summary:
    #         output_feed.append(self.validate_summary)
    #     return session.run(output_feed, input_feed)
   
    
    def predict_step(self, session, data, iter_n, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label(can be empty)
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        #     img (N, w, l, 3)
        #     lidar (N, N', 4)
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional) 
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        doubled_vox_feature = data[5]
        doubled_vox_number = data[6]
        doubled_vox_coordinate = data[7]
        img = data[8]
        lidar = data[9]

        batch_gt_boxes3d = label_to_gt_box3d(label, cls=self.cls, coordinate='lidar')
        print('predict', tag)
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.doubled_vox_feature[idx]] = doubled_vox_feature[idx]
            input_feed[self.doubled_vox_number[idx]] = doubled_vox_number[idx]
            input_feed[self.doubled_vox_coordinate[idx]] = doubled_vox_coordinate[idx]

        output_feed = [self.prob_output, self.delta_output]
        probs, deltas = session.run(output_feed, input_feed)
        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0,1,4,5,6]]
        batch_probs = probs.reshape((len(self.avail_gpus) * self.single_batch_size, -1))
        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(
                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar', check = True))
            ind = session.run(self.box2d_ind_after_nms, {
                self.boxes2d: boxes2d,
                self.boxes2d_scores: tmp_scores
            })
            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

        if summary:
            # only summry 1 in a batch
            for idx in range(len(img)):                
                front_image = draw_lidar_box3d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
                bird_view = lidar_to_bird_view_img(lidar[idx], factor=cfg.BV_LOG_FACTOR)
                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx], factor=cfg.BV_LOG_FACTOR)

                heatmap = colorize(probs[idx, ...], cfg.BV_LOG_FACTOR)

                save_name = "./save_image/{}_{}_fv.png".format(iter_n, tag[idx])
                cv2.imwrite(save_name, front_image)
                save_name = "./save_image/{}_{}_bv.png".format(iter_n, tag[idx])
                cv2.imwrite(save_name, bird_view)
                save_name = "./save_image/{}_{}_hm.png".format(iter_n, tag[idx])
                cv2.imwrite(save_name, heatmap)

            ret_summary = session.run(self.predict_summary, {
                self.rgb: front_image[np.newaxis, ...],
                self.bv: bird_view[np.newaxis, ...],
                self.bv_heatmap: heatmap[np.newaxis, ...]
            })

            return tag, ret_box3d_score, ret_summary

        return tag, ret_box3d_score

    def validate_step(self, session, data, output_path, summary=False, visualize = False):
        # input:  
        #     (N) tag 
        #     (N, N') label(can be empty)
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        #     img (N, w, l, 3)
        #     lidar (N, N', 4)
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional) 
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        doubled_vox_feature = data[5]
        doubled_vox_number = data[6]
        doubled_vox_coordinate = data[7]
        img = data[8]
        lidar = data[9]

        batch_gt_boxes3d = label_to_gt_box3d(label, cls=self.cls, coordinate='lidar')
        # batch_gt_boxes2d = label_to_gt_box2d(label, cls=self.cls, coordinate='camera')
        # warn('validate'.format(tag))
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.doubled_vox_feature[idx]] = doubled_vox_feature[idx]
            input_feed[self.doubled_vox_number[idx]] = doubled_vox_number[idx]
            input_feed[self.doubled_vox_coordinate[idx]] = doubled_vox_coordinate[idx]

        output_feed = [self.prob_output, self.delta_output]
        probs, deltas = session.run(output_feed, input_feed)
        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0,1,4,5,6]]
        batch_probs = probs.reshape((len(self.avail_gpus) * self.single_batch_size, -1))
        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
            # remove box with low score
            ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(
                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar', check = True))
            ind = session.run(self.box2d_ind_after_nms, {
                self.boxes2d: boxes2d,
                self.boxes2d_scores: tmp_scores
            })
            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

        # warn("ret_box3d: {}".format(len(ret_box3d)))
        if visualize == True:
            for idx in range(len(ret_box3d)):
                front_box_2d = draw_lidar_box3d_to_bbox2d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
                front_image = draw_lidar_box3d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
                bird_view = lidar_to_bird_view_img(lidar[idx], factor=cfg.BV_LOG_FACTOR)
                bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx], factor=cfg.BV_LOG_FACTOR)

                heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)

                save_name = "./save_image/valid_{}_2dbbox.png".format(tag[idx])
                cv2.imwrite(save_name, front_box_2d)        
                save_name = "./save_image/valid_{}_fv.png".format(tag[idx])
                cv2.imwrite(save_name, front_image)
                save_name = "./save_image/valid_{}_bv.png".format(tag[idx])
                cv2.imwrite(save_name, bird_view)
                save_name = "./save_image/valid_{}_hm.png".format(tag[idx])
                cv2.imwrite(save_name, heatmap)

        for idx in range(len(ret_box3d)):
            detected_box3d = lidar_to_camera_box(ret_box3d[idx])
            cls = np.array([self.cls for _ in range(len(detected_box3d))])
            scores = ret_score[idx]
            label = box3d_to_label(detected_box3d[np.newaxis, ...], cls[np.newaxis, ...], scores[np.newaxis, ...], include_score = True, coordinate='camera')[0]  # (N')
            # warn("label: {}".format(label))
            f_name = '{}'.format(tag[idx])
            # warn("file name: {}".format(f_name))
            with open(os.path.join(output_path, f_name + '.txt'), 'w+') as f:
                for line in label:
                    f.write(line)
        return tag, ret_box3d_score


    def false_patch_step(self, session, data, output_path, summary=False, visualize = False):
        # input:  
        #     (N) tag 
        #     (N, N') label(can be empty)
        #     vox_feature 
        #     vox_number 
        #     vox_coordinate
        #     img (N, w, l, 3)
        #     lidar (N, N', 4)
        # output: A, B, C
        #     A: (N) tag
        #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
        #     C; summary(optional) 
        tag = data[0]
        label = data[1]
        vox_feature = data[2]
        vox_number = data[3]
        vox_coordinate = data[4]
        doubled_vox_feature = data[5]
        doubled_vox_number = data[6]
        doubled_vox_coordinate = data[7]
        img = data[8]
        lidar = data[9]

        batch_gt_boxes2d = label_to_gt_box2d(label, cls=self.cls, coordinate='camera')

        # warn("tag: {} {}".format(tag[0], tag[1]))
        # warn("gt box 1: {}".format(batch_gt_boxes2d[0]))
        # warn("gt box 2: {}".format(batch_gt_boxes2d[1]))
        # batch_gt_boxes2d = label_to_gt_box2d(label, cls=self.cls, coordinate='camera')
        # warn('validate'.format(tag))
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.vox_feature[idx]] = vox_feature[idx]
            input_feed[self.vox_number[idx]] = vox_number[idx]
            input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
            input_feed[self.doubled_vox_feature[idx]] = doubled_vox_feature[idx]
            input_feed[self.doubled_vox_number[idx]] = doubled_vox_number[idx]
            input_feed[self.doubled_vox_coordinate[idx]] = doubled_vox_coordinate[idx]

        output_feed = [self.prob_output, self.delta_output]
        probs, deltas = session.run(output_feed, input_feed)
        # BOTTLENECK
        batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate='lidar')
        batch_boxes2d = batch_boxes3d[:, :, [0,1,4,5,6]]
        batch_probs = probs.reshape((len(self.avail_gpus) * self.single_batch_size, -1))
        # NMS
        ret_box3d = []
        ret_score = []
        for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
            # remove box with low score
            # ind = np.where(batch_probs[batch_id, :] >= cfg.FALSE_PATCH_THRESH)[0]
            ind = np.where(batch_probs[batch_id, :] >= cfg.FALSE_PATCH_THRESH)[0]

            tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
            tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
            tmp_scores = batch_probs[batch_id, ind]

            # TODO: if possible, use rotate NMS
            boxes2d = corner_to_standup_box2d(
                center_to_corner_box2d(tmp_boxes2d, coordinate='lidar', check = True))
            ind = session.run(self.box2d_ind_after_nms, {
                self.boxes2d: boxes2d,
                self.boxes2d_scores: tmp_scores
            })
            tmp_boxes3d = tmp_boxes3d[ind, ...]
            tmp_scores = tmp_scores[ind]
            ret_box3d.append(tmp_boxes3d)
            ret_score.append(tmp_scores)

        ret_box3d_score = []
        for boxes3d, scores in zip(ret_box3d, ret_score):
            ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
                                                   boxes3d, scores[:, np.newaxis]], axis=-1))

        # warn("ret_box3d: {}".format(len(ret_box3d)))
        # if visualize == True:
        #     for idx in range(len(ret_box3d)):
        #         front_box_2d = draw_lidar_box3d_to_bbox2d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
        #         front_image = draw_lidar_box3d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
        #         bird_view = lidar_to_bird_view_img(lidar[idx], factor=cfg.BV_LOG_FACTOR)
        #         bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx], factor=cfg.BV_LOG_FACTOR)

        #         heatmap = colorize(probs[0, ...], cfg.BV_LOG_FACTOR)

        #         save_name = "./save_image/valid_{}_2dbbox.png".format(tag[idx])
        #         cv2.imwrite(save_name, front_box_2d)        
        #         save_name = "./save_image/valid_{}_fv.png".format(tag[idx])
        #         cv2.imwrite(save_name, front_image)
        #         save_name = "./save_image/valid_{}_bv.png".format(tag[idx])
        #         cv2.imwrite(save_name, bird_view)
        #         save_name = "./save_image/valid_{}_hm.png".format(tag[idx])
        #         cv2.imwrite(save_name, heatmap)

        for idx in range(len(ret_box3d)):
            # detected_box3d = lidar_to_camera_box(ret_box3d[idx])
            # warn("idx: {}".format(idx))
            if len(ret_box3d[idx]) == 0: 
                continue

            projected_bbox2d = lidar_box3d_to_camera_box(ret_box3d[idx], cal_projection=False)

            # warn("proj gt shape: {} {}".format(np.shape(projected_bbox2d), np.shape(batch_gt_boxes2d[idx])))

            if len(batch_gt_boxes2d[idx]) == 0:
                # warn("no gt box all iou 0")
                max_iou = 0
                cls = np.array([self.cls for _ in range(len(projected_bbox2d))])
                scores = ret_score[idx]

                label = box2d_to_label_no_iou(projected_bbox2d[np.newaxis, ...], cls[np.newaxis, ...], scores[np.newaxis, ...])[0]

            else:      
                max_gt_idx = []
                max_ious = []

                for box in projected_bbox2d:
                    max_iou = 0
                    max_gt_id = 0
                    for gt_idx, gt_box in enumerate(batch_gt_boxes2d[idx]):
                        iou = bbox_iou(box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_gt_id = gt_idx
                    max_gt_idx.append(max_gt_id)
                    max_ious.append(max_iou)

                max_gt_idx = np.asarray(max_gt_idx)
                max_ious = np.asarray(max_ious)
                # warn("gt id: {} max_iou : {}".format(max_gt_idx, max_ious))

                cls = np.array([self.cls for _ in range(len(projected_bbox2d))])
                scores = ret_score[idx]
                gt_boxes = batch_gt_boxes2d[idx]
                label = box2d_to_label(projected_bbox2d[np.newaxis, ...], cls[np.newaxis, ...], max_gt_idx[np.newaxis, ...], max_ious[np.newaxis, ...], gt_boxes[np.newaxis, ...], scores[np.newaxis, ...])[0]  # (N')
            # warn("label: {}".format(label))
            f_name = '{}'.format(tag[idx])
            # warn("file name: {}".format(f_name))
            with open(os.path.join(output_path, f_name + '_false.txt'), 'w+') as f:
                for line in label:
                    f.write(line)
        return tag, ret_box3d_score


def average_gradients(tower_grads):
    # ref:  
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads



if __name__ == '__main__':
    pass	
