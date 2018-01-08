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
from model import reduced_vgg

from misc_util import *
# from train_hook import check_if_should_pause

warn("test")

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-i', '--max-epoch', type=int, nargs='?', default=200,
                    help='max epoch')
parser.add_argument('-n', '--tag', type=str, nargs='?', default='default',
                    help='set log tag')
parser.add_argument('-b', '--single-batch-size', type=int, nargs='?', default=60,
                    help='set batch size for each gpu')
parser.add_argument('-l', '--lr', type=float, nargs='?', default=0.001,
                    help='set learning rate')
args = parser.parse_args()

cur_dir = get_cur_dir()

dataset_dir = os.path.join(cur_dir, 'data/object')
warn("dataset_dir: {}".format(dataset_dir))
# dataset_dir = '../data/object'
log_dir = os.path.join('./log2d', args.tag)
save_model_dir = os.path.join('./save_model_2d', args.tag)
save_image_dir = os.path.join('./save_image_2d', args.tag)
save_result_dir = './result_2d'
mkdir_p(log_dir)
mkdir_p(save_model_dir)
mkdir_p(save_image_dir)
mkdir_p(save_result_dir)


dataset = 'training'
from utils.kitti_2dloader import KittiLoader
split_file = 'trainset.txt'
valid_file = 'validset.txt'

def main(_):
    # TODO: split file support
    warn("main start")
    with tf.Graph().as_default():
        global save_model_dir
        with KittiLoader(object_dir=os.path.join(dataset_dir, dataset), queue_size=200, require_shuffle=True, 
                is_testset=False, batch_size=args.single_batch_size*cfg.GPU_USE_COUNT, use_multi_process_num=6, split_file = split_file, multi_gpu_sum=cfg.GPU_USE_COUNT) as train_loader, \
            KittiLoader(object_dir=os.path.join(dataset_dir, dataset), queue_size=200, require_shuffle=True, 
                is_testset=False, batch_size=args.single_batch_size*cfg.GPU_USE_COUNT, use_multi_process_num=6, split_file = valid_file, multi_gpu_sum=cfg.GPU_USE_COUNT) as valid_loader:
            
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
                model = reduced_vgg(
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
                validate_interval = 1000

                # if is_test == True:
                #     for test_idx in range(5236, train_loader.dataset_size):
                #         t0 = time.time()
                #         ret = model.test_step(sess, train_loader.load_specified(test_idx), output_path = save_result_dir, summary=True)
                #         t1 = time.time()
                #         warn("test: {:.2f}".format(t1-t0))
                
                t0 = time.time()
                summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
                while model.epoch.eval() < args.max_epoch:
                    is_summary, is_summary_image, is_validate = False, False, False 
                    progress = model.epoch.eval()/args.max_epoch
                    train_loader.progress = progress

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
                    # t0 = time.time()
                    ret = model.train_step(sess, train_loader.load(), train=True, summary=is_summary)
                    # t1 = time.time()
                    # warn("train: {}".format(t1-t0))
                    if iter % 100 == 0:
                        avg_time = (time.time() - t0)/100
                        t0 = time.time()
                        print('train: {} / 1 : {}/{} @ epoch:{}/{} {:.2f} sec, remaining: {:.2f} min, loss: {:.2f} {}'.format(train_loader.progress, iter, 
                            iter_per_epoch*args.max_epoch, model.epoch.eval(), args.max_epoch, avg_time, avg_time*(iter_per_epoch*args.max_epoch - iter)//60, ret[0], args.tag))

                    if is_summary:
                        summary_writer.add_summary(ret[-1], iter)

                    # if is_summary_image:
                    #     t0 = time.time()
                    #     ret = model.predict_step(sess, train_loader.load_specified(), iter, summary=True)
                    #     summary_writer.add_summary(ret[-1], iter)
                    #     t1= time.time()
                    #     warn("predict: {}".format(t1-t0))

                    if is_validate:   
                        warn("validation: {}".format(iter))                     
                        total_iter = int(np.ceil(valid_loader.dataset_size / valid_loader.batch_size))
                        # idx = iter
                        save_result_folder = os.path.join(save_result_dir, "{}".format(iter))
                        mkdir_p(save_result_folder)
                        total_loss = 0
                        for idx in range(total_iter):
                            t0 = time.time()
                            loss = model.validate_step(sess, valid_loader.load(), output_path= save_result_folder, summary=True, visualize = False)
                            t1= time.time()
                            total_loss = total_loss + loss
                            # warn("valid: {:.2f} sec | remaining {:.2f} sec {}/{}".format(t1-t0, (t1-t0)*(total_iter-idx), idx, total_iter))
                        warn("validation: {} epoch: total loss sum: {}".format(model.epoch.eval(), total_loss))
                    #     cmd = "./evaluate_object {}".format(iter)
                    #     os.system(cmd)

                        
                        # warn("test: {}".format(t1-t0))
                    # if is_validate:
                    #     ret = model.test_step(sess, train_loader.load_specified(), summary=True)
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
