#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import glob
import argparse
import os
import time
import sys
import tensorflow as tf
from itertools import count
from misc_util import get_cur_dir, warn, mkdir_p
import cv2
from utils.utils import label_to_gt_box2d, bbox_iou, random_distort_image, draw_bbox2d_on_image
import numpy as np
from config import cfg
# from data_aug import image_augmentation

cur_dir = get_cur_dir()

dataset_dir = os.path.join(cur_dir, 'data/object')
warn("dataset_dir: {}".format(dataset_dir))
dataset = 'training'
split_file = 'trainset.txt'

test_img_save_dir = 'test_img'
test_img_save_dir = os.path.join(cur_dir, test_img_save_dir)
mkdir_p(test_img_save_dir)

def image_augmentation(f_rgb, f_label, width, height, jitter, hue, saturation, exposure):
	rgb_imgs = []
	ious = []
	org_imgs = []
	label = np.array([line for line in open(f_label, 'r').readlines()])
	gt_box2d = label_to_gt_box2d(np.array(label)[np.newaxis, :], cls=cfg.DETECT_OBJ, coordinate='lidar')[0]  # (N', 4) x_min, y_min, x_max, y_max

	img = cv2.imread(f_rgb)
	warn("img value: {}".format(img[:3,:3,:3]))

	# warn("{} shape: {}".format(f_rgb, img.shape))
	img_height, img_width = img.shape[:2]
	# warn("height: {}, width: {}".format(img_height, img_width))

	for idx in range(len(gt_box2d)):
		box = gt_box2d[idx]
		# warn("box {}: {}".format(idx, box))
		x_min, y_min, x_max, y_max = box
		x_min = int(x_min)
		y_min = int(y_min)
		x_max = int(x_max)
		y_max = int(y_max)

		ori_img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max, x_min:x_max], (64, 64))
		org_imgs.append(ori_img)

		box_height = y_max - y_min
		box_width = x_max - x_min

		dx = int(jitter * box_width) + 1
		dy = int(jitter * box_height) + 1

		# warn("dx : {} dy : {}".format(dx, dy))

		lx = np.random.randint(-dx, dx)
		ly = np.random.randint(-dy, dy)

		lw = np.random.randint(-dx, dx)
		lh = np.random.randint(-dy, dy)

		x = (x_max + x_min)/2.0 + lx
		y = (y_max + y_min)/2.0 + ly
		box_height = box_height + lh
		box_width = box_width + lw

		x_min = int(max(0, x - box_width/2.0))
		x_max = int(min(img_width, x + box_width/2.0))
		y_min = int(max(0, y - box_height/2.0))
		y_max = int(min(img_height, y + box_height/2.0))


		flip = np.random.randint(1,10000)%2  

		img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max,x_min:x_max], (width, height))

		if flip:
		    img = cv2.flip(img, 1)
		img = random_distort_image(img, hue, saturation, exposure)
		# for ground truth img, calculate iou with its original location, size

		iou = bbox_iou(box, (x_min, y_min, x_max, y_max), x1y1x2y2=True)


		rgb_imgs.append(img)
		ious.append(iou)



	# Randomly e[nerate same number of background candidate that will have low iou or zero iou.
	# after generating new boxes, it needs to calculate iou to each of gt_boxes2d 
	# which will be used as inference.
	# if inferenced iou is low, then the bounding boxes are empty or background or falsely located.
	# if inferenced iou is high, then the bounding boxes are correctly inferenced by 3D bounding boxes.
	# this is the st]rategry I am taking for simple, mini 2D classifier.

	for idx in range(len(gt_box2d)*4):
		x = np.random.randint(0, img_width)
		y = np.random.randint(0, img_height)
		h = np.random.randint(40, 200)
		w = np.random.randint(40, 200)
		x_min = int(max(0, x - w/2.0))
		x_max = int(min(img_width, x + w/2.0))
		y_min = int(max(0, y - h/2.0))
		y_max = int(min(img_height, y + h/2.0))

		max_iou = 0

		for gt_idx in range(len(gt_box2d)):
			box = gt_box2d[gt_idx]
			iou = bbox_iou(box, (x_min, y_min, x_max, y_max), x1y1x2y2=True)
			if iou > max_iou:
				max_iou = iou

		img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max,x_min:x_max], (width, height))
		if flip:
			img = cv2.flip(img, 1)
		img = random_distort_image(img, hue, saturation, exposure)
		rgb_imgs.append(img)
		ious.append(iou)


	return org_imgs, rgb_imgs, ious

def main():

	object_dir = os.path.join(dataset_dir, dataset)
	f_rgb, f_label, f_line = [], [], []
	for line in open(split_file, 'r').readlines():
		line = line[:-1] # remove '\n'
		f_rgb.append(os.path.join(object_dir, 'image_2', line+'.png'))
		f_label.append(os.path.join(object_dir, 'label_2', line+'.txt'))
		f_line.append(line)

	width, height = 64, 64
	max_error = 0.1

	jitter = 0.1
	hue = 0.1#0.1
	saturation = 1.5#1.5 
	exposure = 1.5#1.5


	for load_index in range(1,5):#range(len(f_rgb)):
		warn("{} / {}".format(load_index, len(f_rgb)))
		img = cv2.imread(f_rgb[load_index])
		save_dir = os.path.join(dataset_dir, dataset)
		org_imgs, cropped_img, ious = image_augmentation(f_rgb[load_index], f_label[load_index], width, height, jitter, hue, saturation, exposure)
		save_file = os.path.join(test_img_save_dir, '{}.png'.format(f_line[load_index]))

		label = np.array([line for line in open(f_label[load_index], 'r').readlines()])
		gt_box2d = label_to_gt_box2d(np.array(label)[np.newaxis, :], cls=cfg.DETECT_OBJ, coordinate='lidar')[0]  # (N', 4) x_min, y_min, x_max, y_max

		img = draw_bbox2d_on_image(img, gt_box2d)

		cv2.imwrite(save_file, img)
		for index in range(len(cropped_img)):
			save_file = os.path.join(test_img_save_dir, '{}_{}.png'.format(f_line[load_index], index))
			cv2.imwrite(save_file, cropped_img[index])
			# save_file = os.path.join(test_img_save_dir, '{}_{}_ori.png'.format(f_line[load_index], index))
			# cv2.imwrite(save_file, org_imgs[index])


if __name__ == '__main__':
	warn("main start")
	main()