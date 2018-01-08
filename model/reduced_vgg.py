#!/usr/bin/env python
# -*- coding:UTF-8 -*-
import sys
import os
import tensorflow as tf

from config import cfg
from utils.utils import * 
from model.group_pointcloud import FeatureNet, FeatureNet_new
from model.rpn import MiddleAndRPN, ConvMD
from misc_util import warn
from utils.colorize import *

VGG_MEAN = [103.939, 116.779, 123.68]
small_addon_for_BCE=1e-6

class vgg(object):

    def __init__(self, training, batch_size, name=''):
        super(vgg, self).__init__()
        self.training = training

        # scalar
        self.batch_size = batch_size
        # [ΣK, 64, 64, 3]
        self.imgs = tf.placeholder(tf.float32, [None, 64, 64, 3], name='img')
        self.conf = tf.placeholder(tf.float32, [None], name='conf')
        start_time = time.time()
        print("build model started")
        # rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.imgs)
        assert red.get_shape().as_list()[1:] == [64, 64, 1]
        assert green.get_shape().as_list()[1:] == [64, 64, 1]
        assert blue.get_shape().as_list()[1:] == [64, 64, 1]

        x = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
            temp_conv = ConvMD(2, 3, 16, 3, (1,1), (1,1), x, training = self.training, name = 'conv1_1')
            # => [None, 64, 64, 16]
            temp_conv = ConvMD(2, 16, 16, 3, (1,1), (1,1), temp_conv, training = self.training, name = 'conv1_2')
            # => [None, 64, 64, 16]
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool1')
            # => [None, 32, 32, 16]
            temp_conv = ConvMD(2, 16, 16, 3, (1,1), (1,1), temp_conv, training = self.training, name = 'conv2_1')
            # => [None, 32, 32, 16]
            temp_conv = ConvMD(2, 16, 16, 3, (1,1), (1,1), temp_conv, training = self.training, name = 'conv2_2')
            # => [None, 32, 32, 16]
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool2')
            # => [None, 16, 16, 16]
            temp_conv = ConvMD(2, 16, 16, 3, (1,1), (1,1), temp_conv, training = self.training, name = 'conv3_1')
            # => [None, 16, 16, 16]
            temp_conv = ConvMD(2, 16, 16, 3, (1,1), (1,1), temp_conv, training = self.training, name = 'conv3_2')
            # => [None, 16, 16, 16]
            temp_conv = tf.layers.max_pooling2d(temp_conv, pool_size = 2, strides = 2, name = 'maxpool3')
            # => [None, 8, 8, 16]
            warn("shape: {}".format(np.shape(temp_conv)))
            temp = tf.layers.flatten(temp_conv, name='flatten')
            # => [None, 64 * 16]
            warn("shape: {}".format(np.shape(temp)))
            temp = tf.nn.relu(self.dense(temp, 32, 'dense1'))
            warn("shape: {}".format(np.shape(temp)))
            self.prob = tf.nn.sigmoid(self.dense(temp, 1, 'prob'))
            warn("shape: {}".format(np.shape(self.prob)))
            self.loss = tf.reduce_mean(-self.conf * tf.log(self.prob+small_addon_for_BCE) - (1 - self.conf) * tf.log(1-self.prob+small_addon_for_BCE))
            # self.loss = tf.nn.l2_loss(self.conf - self.prob, name = 'loss')

    def dense(self, x, size, name):
        w = tf.get_variable(name+'/w', [x.get_shape()[1], size], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name+'/b', [size], initializer=tf.zeros_initializer())
        return tf.matmul(x,w) + b

    # def normc_initializer(std=1.0):
    #     def _initializer(shape, dtype=None, partition_info=None):
    #         out = np.random.randn(*shape).astype(np.float32)
    #         out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
    #         return tf.constant(out)
    #     return _initializer

class reduced_vgg(object):

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


        self.imgs = []
        self.confs = [] 
        # => wip: add confidence(iou) here for yolo style
        # => pos_equal_one is actually conf_mask in yolo code
        # self.conf_target = []

        self.prob_output = []
        self.opt = tf.train.AdamOptimizer(lr)
        self.gradient_norm = []
        self.tower_grads = []
        self.batch_loss = []
        with tf.variable_scope(tf.get_variable_scope()):
            for idx, dev in enumerate(self.avail_gpus):
                with tf.device('/gpu:{}'.format(dev)), tf.name_scope('gpu_{}'.format(dev)):
                    # must use name scope here since we do not want to create new variables
                    # graph
                    vggnet= vgg(training=is_train, batch_size=self.single_batch_size, name = 'vgg')

                    tf.get_variable_scope().reuse_variables()

                    # input
                    self.imgs.append(vggnet.imgs)
                    self.confs.append(vggnet.conf)

                    # output
                    prob_output = vggnet.prob

                    # loss and grad
                    self.loss = vggnet.loss
                    self.params = tf.trainable_variables()
                    gradients = tf.gradients(self.loss, self.params)
                    clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

                    self.prob_output.append(prob_output)
                    self.tower_grads.append(clipped_gradients)
                    self.gradient_norm.append(gradient_norm)
                    self.batch_loss.append(self.loss)

        # loss and optimizer
        # self.xxxloss is only the loss for the lowest tower
        with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
            self.grads = average_gradients(self.tower_grads)
            self.update = self.opt.apply_gradients(zip(self.grads, self.params), global_step=self.global_step)
            self.gradient_norm = tf.group(*self.gradient_norm)

        self.prob_output = tf.concat(self.prob_output, axis=0)

        warn("batch loss1: {}".format(np.shape(self.batch_loss)))
        self.batch_loss = tf.reduce_sum(self.batch_loss)
        warn("batch loss2: {}".format(np.shape(self.batch_loss)))

        # # for predict and image summary 
        # self.rgb = tf.placeholder(tf.uint8, [None, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH, 3])


        # self.bv = tf.placeholder(tf.uint8, [
        #                          None, cfg.BV_LOG_FACTOR * cfg.INPUT_HEIGHT, cfg.BV_LOG_FACTOR * cfg.INPUT_WIDTH, 3])
        # self.bv_heatmap = tf.placeholder(tf.uint8, [
        #     None, cfg.BV_LOG_FACTOR * cfg.FEATURE_HEIGHT, cfg.BV_LOG_FACTOR * cfg.FEATURE_WIDTH, 3])
        # self.boxes2d = tf.placeholder(tf.float32, [None, 4])
        # self.boxes2d_scores = tf.placeholder(tf.float32, [None])


        # # NMS(2D)
        # with tf.device('/gpu:{}'.format(self.avail_gpus[0])):
        #     self.box2d_ind_after_nms = tf.image.non_max_suppression(self.boxes2d, self.boxes2d_scores, max_output_size=cfg.RPN_NMS_POST_TOPK, iou_threshold=cfg.RPN_NMS_THRESH)
    
        # summary and saver
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

        self.train_summary = tf.summary.merge([
            tf.summary.scalar('train/loss', self.loss),
            # tf.summary.scalar('train/reg_loss', self.reg_loss),
            # tf.summary.scalar('train/cls_loss', self.cls_loss),
            [tf.summary.histogram(each.name, each) for each in self.params]
        ])

        self.validate_summary = tf.summary.merge([
            tf.summary.scalar('validate/loss', self.loss)
        ])

        # # TODO: bird_view_summary and front_view_summary
        
        # self.predict_summary = tf.summary.merge([
        #     tf.summary.image('predict/bird_view_lidar', self.bv),
        #     tf.summary.image('predict/bird_view_heatmap', self.bv_heatmap),
        #     tf.summary.image('predict/front_view_rgb', self.rgb),
        # ])


    def train_step(self, session, data, train=False, summary=False):
        # input:  
        #     (N) tag 
        #     (N, N') label
        imgs = data[0]
        confs = data[1]
        # warn("shape: {} {}".format(np.shape(imgs), np.shape(confs))) 
        confs[confs>=0.6] = 1.0
        confs[confs<0.6] = 0.0
        
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.imgs[idx]] = imgs[idx]
            input_feed[self.confs[idx]] = confs[idx]
        if train:
            output_feed = [self.batch_loss, self.gradient_norm, self.update]
        else:
            output_feed = [self.batch_loss]
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
    #     imgs = data[0]
    #     confs = data[1]

    #     warn("validating")
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
   
    
    # def predict_step(self, session, data, iter_n, summary=False):
    #     # input:  
    #     #     (N) tag 
    #     #     (N, N') label(can be empty)
    #     #     vox_feature 
    #     #     vox_number 
    #     #     vox_coordinate
    #     #     img (N, w, l, 3)
    #     #     lidar (N, N', 4)
    #     # output: A, B, C
    #     #     A: (N) tag
    #     #     B: (N, N') (class, x, y, z, h, w, l, rz, score)
    #     #     C; summary(optional) 
    #     tag = data[0]
    #     label = data[1]
    #     vox_feature = data[2]
    #     vox_number = data[3]
    #     vox_coordinate = data[4]
    #     doubled_vox_feature = data[5]
    #     doubled_vox_number = data[6]
    #     doubled_vox_coordinate = data[7]
    #     img = data[8]
    #     lidar = data[9]

    #     batch_gt_boxes3d = label_to_gt_box3d(label, cls=self.cls, coordinate='lidar')
    #     print('predict', tag)
    #     input_feed = {}
    #     for idx in range(len(self.avail_gpus)):
    #         input_feed[self.vox_feature[idx]] = vox_feature[idx]
    #         input_feed[self.vox_number[idx]] = vox_number[idx]
    #         input_feed[self.vox_coordinate[idx]] = vox_coordinate[idx]
    #         input_feed[self.doubled_vox_feature[idx]] = doubled_vox_feature[idx]
    #         input_feed[self.doubled_vox_number[idx]] = doubled_vox_number[idx]
    #         input_feed[self.doubled_vox_coordinate[idx]] = doubled_vox_coordinate[idx]

    #     output_feed = [self.prob_output, self.delta_output]
    #     probs, deltas = session.run(output_feed, input_feed)
    #     # BOTTLENECK
    #     batch_boxes3d = delta_to_boxes3d(deltas, self.anchors, coordinate='lidar')
    #     batch_boxes2d = batch_boxes3d[:, :, [0,1,4,5,6]]
    #     batch_probs = probs.reshape((len(self.avail_gpus) * self.single_batch_size, -1))
    #     # NMS
    #     ret_box3d = []
    #     ret_score = []
    #     for batch_id in range(len(self.avail_gpus) * self.single_batch_size):
    #         # remove box with low score
    #         ind = np.where(batch_probs[batch_id, :] >= cfg.RPN_SCORE_THRESH)[0]
    #         tmp_boxes3d = batch_boxes3d[batch_id, ind, ...]
    #         tmp_boxes2d = batch_boxes2d[batch_id, ind, ...]
    #         tmp_scores = batch_probs[batch_id, ind]

    #         # TODO: if possible, use rotate NMS
    #         boxes2d = corner_to_standup_box2d(
    #             center_to_corner_box2d(tmp_boxes2d, coordinate='lidar', check = True))
    #         ind = session.run(self.box2d_ind_after_nms, {
    #             self.boxes2d: boxes2d,
    #             self.boxes2d_scores: tmp_scores
    #         })
    #         tmp_boxes3d = tmp_boxes3d[ind, ...]
    #         tmp_scores = tmp_scores[ind]
    #         ret_box3d.append(tmp_boxes3d)
    #         ret_score.append(tmp_scores)

    #     ret_box3d_score = []
    #     for boxes3d, scores in zip(ret_box3d, ret_score):
    #         ret_box3d_score.append(np.concatenate([np.tile(self.cls, len(boxes3d))[:, np.newaxis],
    #                                                boxes3d, scores[:, np.newaxis]], axis=-1))

    #     if summary:
    #         # only summry 1 in a batch
    #         for idx in range(len(img)):                
    #             front_image = draw_lidar_box3d_on_image(img[idx], ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx])
    #             bird_view = lidar_to_bird_view_img(lidar[idx], factor=cfg.BV_LOG_FACTOR)
    #             bird_view = draw_lidar_box3d_on_birdview(bird_view, ret_box3d[idx], ret_score[idx], batch_gt_boxes3d[idx], factor=cfg.BV_LOG_FACTOR)

    #             heatmap = colorize(probs[idx, ...], cfg.BV_LOG_FACTOR)

    #             save_name = "./save_image/{}_{}_fv.png".format(iter_n, tag[idx])
    #             cv2.imwrite(save_name, front_image)
    #             save_name = "./save_image/{}_{}_bv.png".format(iter_n, tag[idx])
    #             cv2.imwrite(save_name, bird_view)
    #             save_name = "./save_image/{}_{}_hm.png".format(iter_n, tag[idx])
    #             cv2.imwrite(save_name, heatmap)

    #         ret_summary = session.run(self.predict_summary, {
    #             self.rgb: front_image[np.newaxis, ...],
    #             self.bv: bird_view[np.newaxis, ...],
    #             self.bv_heatmap: heatmap[np.newaxis, ...]
    #         })

    #         return tag, ret_box3d_score, ret_summary

    #     return tag, ret_box3d_score

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

        imgs = data[0]
        confs = data[1]
        # warn("shape: {} {}".format(np.shape(vox_feature), np.shape(doubled_vox_feature))) 

        confs[confs>=0.5] = 1.0
        confs[confs<0.5] = 0.0
        
        input_feed = {}
        for idx in range(len(self.avail_gpus)):
            input_feed[self.imgs[idx]] = imgs[idx]
            input_feed[self.confs[idx]] = confs[idx]

        output_feed = [self.batch_loss, self.prob_output]

        if summary:
            output_feed.append(self.train_summary)
        # TODO: multi-gpu support for test and predict step
        batch_loss, prob_output, _ =  session.run(output_feed, input_feed)

        prob_output = np.squeeze(prob_output)

        # warn("conf: {}".format(np.shape(confs)))
        prob_output[prob_output>=0.5] = 1.0
        prob_output[prob_output<0.5] = 0.0
        # warn("prob: {}".format(np.shape(prob_output)))

        confs = np.concatenate(confs, axis=0)
        # warn("conf: {}".format(np.shape(confs)))

        correct_precision = np.equal(confs, prob_output)
        correct_precision = correct_precision.astype(int)
        # warn("result: {}".format(correct_precision))

        gt_neg = (confs == 0.0)
        gt_neg = np.sum(gt_neg.astype(int))
        gt_pos = (confs == 1.0)
        gt_pos = np.sum(gt_pos.astype(int))
        true_pos = np.logical_and((confs == 1.0), (prob_output == 1.0))
        true_neg = np.logical_and((confs == 0.0), (prob_output == 0.0))
        false_pos = np.logical_and((confs == 0.0), (prob_output == 1.0))
        false_neg = np.logical_and((confs == 1.0), (prob_output == 0.0))
        true_pos = np.sum(true_pos.astype(int))
        true_neg = np.sum(true_neg.astype(int))
        false_pos = np.sum(false_pos.astype(int))
        false_neg = np.sum(false_neg.astype(int))


        return batch_loss, gt_pos, gt_neg, true_pos, true_neg, false_pos, false_neg


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
