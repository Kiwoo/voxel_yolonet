#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import sys
import tensorflow as tf
from itertools import count

from config import cfg
from model import RPN3D
from utils.kitti_loader import KittiLoader
from misc_util import *
# from train_hook import check_if_should_pause

warn("test")

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=10,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=1,
                    help='set batch size for each gpu')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
args = parser.parse_args()

cur_dir = get_cur_dir()

dataset_dir = os.path.join(cur_dir, 'data/object')
warn("dataset_dir: {}".format(dataset_dir))
# dataset_dir = '../data/object'
log_dir = os.path.join('./log', args.tag)
save_model_dir = os.path.join('./save_model', args.tag)
mkdir_p(log_dir)
mkdir_p(save_model_dir)

k = os.path.join(dataset_dir, 'training')
warn("dir: {}".format(k))

def main(_):
    # TODO: split file support
    warn("main start")
    with tf.Graph().as_default():
        global save_model_dir
        with KittiLoader(object_dir=os.path.join(dataset_dir, 'training'), queue_size=5, require_shuffle=True, 
                is_testset=False, batch_size=args.single_batch_size*cfg.GPU_USE_COUNT, use_multi_process_num=2, multi_gpu_sum=cfg.GPU_USE_COUNT) as train_loader:
        # , \KittiLoader(object_dir=os.path.join(dataset_dir, 'testing'), queue_size=50, require_shuffle=True, 
        #         is_testset=False, batch_size=args.single_batch_size*cfg.GPU_USE_COUNT, use_multi_process_num=8, multi_gpu_sum=cfg.GPU_USE_COUNT) as valid_loader:
            
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.GPU_MEMORY_FRACTION,
                                        visible_device_list=cfg.GPU_AVAILABLE,
                                        allow_growth=True)
            config = tf.ConfigProto(
                gpu_options=gpu_options,
                device_count={
                    "GPU": cfg.GPU_USE_COUNT,
                },
                allow_soft_placement=True,
            )
            # tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
            # tf_config.gpu_options.allow_growth = True
            warn("1")
            with tf.Session(config=config) as sess:
                model = RPN3D(
                    cls=cfg.DETECT_OBJ,
                    single_batch_size=args.single_batch_size,
                    learning_rate=args.lr,
                    max_gradient_norm=5.0,
                    is_train=True,
                    alpha=1.5,
                    beta=1,
                    avail_gpus=cfg.GPU_AVAILABLE.split(',')
                )
                warn("2")
                # param init/restore
                if tf.train.get_checkpoint_state(save_model_dir):
                    warn("Reading model parameters from %s" % save_model_dir)
                    model.saver.restore(sess, tf.train.latest_checkpoint(save_model_dir))
                else:
                    warn("Created model with fresh parameters.")
                    tf.global_variables_initializer().run()

                # train and validate
                iter_per_epoch = int(len(train_loader)/(args.single_batch_size*cfg.GPU_USE_COUNT))
                is_summary, is_summary_image, is_validate = False, False, False 
                
                summary_interval = 50
                summary_image_interval = 50
                save_model_interval = 50
                validate_interval = 60
                
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                while model.epoch.eval() < args.max_epoch:
                    is_summary, is_summary_image, is_validate = False, False, False 
                    iter = model.global_step.eval()
                    if not iter % summary_interval:
                        is_summary = True
                    if not iter % summary_image_interval:
                        is_summary_image = True 
                    if not iter % save_model_interval:
                        model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
                    if not iter % validate_interval:
                        is_validate = True
                    if not iter % iter_per_epoch:
                        sess.run(model.epoch_add_op)
                        t1 = time.time()
                        warn("train: {}".format(t1-t0))
                        print('train {} epoch, total: {}'.format(model.epoch.eval(), args.max_epoch))
                    t0 = time.time()
                    ret = model.train_step(sess, train_loader.load(), train=True, summary=is_summary)
                    t1 = time.time()
                    warn("train: {}".format(t1-t0))
                    print('train: {}/{} @ epoch:{}/{} {:.2f} sec, loss: {} reg_loss: {} cls_loss: {} {}'.format(iter, 
                        iter_per_epoch*args.max_epoch, model.epoch.eval(), args.max_epoch, t1-t0, ret[0], ret[1], ret[2], args.tag))

                    if is_summary:
                        summary_writer.add_summary(ret[-1], iter)

                    if is_summary_image:
                        t0 = time.time()
                        ret = model.predict_step(sess, train_loader.load(), iter, summary=True)
                        summary_writer.add_summary(ret[-1], iter)
                        t1= time.time()
                        warn("predict: {}".format(t1-t0))

                    # if is_validate:
                    #     ret = model.validate_step(sess, valid_loader.load(), summary=True)
                    #     summary_writer.add_summary(ret[-1], iter)
                    
                    # if check_if_should_pause(args.tag):
                    #     model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)
                    #     print('pause and save model @ {} steps:{}'.format(save_model_dir, model.global_step.eval()))
                    #     sys.exit(0)

                print('train done. total epoch:{} iter:{}'.format(model.epoch.eval(), model.global_step.eval()))
                
                # finallly save model
                model.saver.save(sess, os.path.join(save_model_dir, 'checkpoint'), global_step=model.global_step)

if __name__ == '__main__':
    warn("main start")
    tf.app.run(main)
