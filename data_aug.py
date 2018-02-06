#!/usr/bin/env python
# -*- coding:UTF-8 -*-

# File Name : data_aug.py
# Purpose :
# Creation Date : 21-12-2017
# Last Modified : Sat 23 Dec 2017 11:58:13 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import cv2
import os
import multiprocessing as mp
import argparse
import glob

from utils.utils import *
from utils.preprocess import clip_by_projection

object_dir = './data/object'
output_path = os.path.join(object_dir, 'training')

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--aug-amount', type=int, nargs='?', default=1000)
parser.add_argument('-n', '--num-workers', type=int, nargs='?', default=10)
args = parser.parse_args()

def data_augmentation(f_lidar, f_label, calib_mat, img_width, img_height):

    t0 = time.time()
    shift_x = [-5.0, 5.0]
    shift_y = [-5.0, 5.0]
    shift_z = [-0.3, 0.3]
    angle_r = [-np.pi/4, np.pi/4]
    scale = [0.95, 1.05] 
    drop_rate = [0.01, 0.15]

    lidar = np.fromfile(f_lidar, dtype=np.float32).reshape((-1, 4))

    calib_file = f_lidar.replace('velodyne', 'calib').replace('bin', 'txt')

    lidar = clip_by_projection(lidar, calib_file, img_height, img_width)


    label = np.array([line for line in open(f_label, 'r').readlines()])
    cls = np.array([line.split()[0] for line in label])  # (N')
    # warn("shape: {}".format(np.shape(calib_mat[0])))
    calib_mats = []
    calib_mats.append(calib_mat)
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='lidar', calib_mats=calib_mats)[
        0]  # (N', 7) x, y, z, h, w, l, r
# warn("lidar: {} lable: {} choice: {}".format(np.shape(lidar), label, choice))

    # Random drop out every time
    drop = np.random.uniform(drop_rate[0], drop_rate[1])
    num_points = len(lidar)
    np.random.shuffle(lidar)
    end_points = int(num_points * (1 - drop))

    lidar = lidar[:end_points]

    choice = np.random.randint(1, 10)

    if choice >= 8:
        # global shift, rotation
        t_x = np.random.uniform(shift_x[0], shift_x[1])
        t_y = np.random.uniform(shift_y[0], shift_y[1])
        angle = np.random.uniform(angle_r[0], angle_r[1])

        t_z = np.random.uniform(shift_z[0], shift_z[1])

        lidar[:, 0:3] = point_transform(lidar[:, 0:3], tx=t_x, ty=t_y, tz=t_z, rx=0, ry=0, rz=angle)

        lidar_center_gt_box3d = box_transform(gt_box3d, tx=t_x, ty=t_y, tz=t_z, r=angle, coordinate='lidar', calib_mat=calib_mat)
        # gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, calib_mat=calib_mat)

        factor = np.random.uniform(scale[0], scale[1])
        lidar[:, 0:3] = lidar[:, 0:3] * factor
        lidar_center_gt_box3d[:, 0:6] = lidar_center_gt_box3d[:, 0:6] * factor
        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, calib_mat=calib_mat)

    else:#if choice == 1:
        # local shift, rotation
        lidar_center_gt_box3d = gt_box3d
        lidar_corner_gt_box3d = center_to_corner_box3d(
            lidar_center_gt_box3d, coordinate='lidar', calib_mat=calib_mat)
        changed = False
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            is_collision = True
            _count = 0
            # warn("check :{}".format(idx))
            while is_collision and _count < 100:
                t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
                t_x = 5*np.random.normal()
                t_y = 5*np.random.normal()
                t_z = np.random.normal()
                factor = np.random.uniform(scale[0], scale[1])
                # check collision
                # warn("shape: {}".format(np.shape(lidar_center_gt_box3d[[idx]])))
                tmp = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar', calib_mat=calib_mat)
                tmp[0][0:6] = tmp[0][0:6] * factor
                is_collision = False
                for idy in range(idx):
                    x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                    x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
                        0, 1, 4, 5, 6]]
                    iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                    np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                    if iou > 0:
                        is_collision = True
                        _count += 1
                        break
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                lidar_center_gt_box3d[idx] = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar', calib_mat=calib_mat)
                changed = True
        # if changed == True and len(lidar_center_gt_box3d) == 1:
        #     warn("changed: {} {} {} {} {} -> {}".format(len(lidar_center_gt_box3d), t_x, t_y, t_z, t_rz, lidar_center_gt_box3d[0]))
        # else:
        #     warn("unchanged")

        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, calib_mat=calib_mat)
        # warn("gt shape after: {}".format(np.shape(gt_box3d)))

    # else:
    #     # global scaling
    #     factor = np.random.uniform(scale[0], scale[1])
    #     lidar[:, 0:3] = lidar[:, 0:3] * factor
    #     lidar_center_gt_box3d[:, 0:6] = gt_box3d[:, 0:6] * factor
    #     gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d, calib_mat=calib_mat)

    # To clip after rotation and translation

    # lidar = clip_by_projection(lidar, calib_file, cfg.IMAGE_HEIGHT, cfg.IMAGE_WIDTH)
    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], calib_mats, coordinate='camera')[0]  # (N')
    # warn("end: lidar: {} label: {}".format(np.shape(lidar), np.shape(label)))
    return lidar, label

def image_augmentation(f_rgb, f_label, width, height, jitter, hue, saturation, exposure, posneg_ratio):
    rgb_imgs = []
    confs = []
    org_imgs = []
    label = np.array([line for line in open(f_label, 'r').readlines()])
    gt_box2d = label_to_gt_box2d(np.array(label)[np.newaxis, :], cls=cfg.DETECT_OBJ, coordinate='lidar')[0]  # (N', 4) x_min, y_min, x_max, y_max

    img = cv2.imread(f_rgb)

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

        # ori_img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max, x_min:x_max], (64, 64))
        # org_imgs.append(ori_img)

        box_height = y_max - y_min
        box_width = x_max - x_min

        if box_height < 25 or box_width < 25:
            continue

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

        try:
            img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max,x_min:x_max], (width, height))
        except:
            warn("1 {} {} {} {} {} {} {} {} {} {} {}".format(f_rgb, dx, dy, lx, ly, y_min, y_max, x_min, x_max, width, height))

        if flip:
            img = cv2.flip(img, 1)
        img = random_distort_image(img, hue, saturation, exposure)
        # for ground truth img, calculate iou with its original location, size

        iou = bbox_iou(box, (x_min, y_min, x_max, y_max), x1y1x2y2=True)


        rgb_imgs.append(img)
        confs.append(iou)

    # after generating new boxes, it needs to calculate iou to each of gt_boxes2d 
    # which will be used as inference.
    # if inferenced iou is low, then the bounding boxes are empty or background or falsely located.
    # if inferenced iou is high, then the bounding boxes are correctly inferenced by 3D bounding boxes.
    # this is the st]rategry I am taking for simple, mini 2D classifier.

    for idx in range(len(gt_box2d)*posneg_ratio):
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

        try:
            img = cv2.resize(cv2.imread(f_rgb)[y_min:y_max,x_min:x_max], (width, height))
        except:
            warn("2 {} {} {} {} {} {} {} {} {} {} {}".format(f_rgb, dx, dy, lx, ly, y_min, y_max, x_min, x_max, width, height))
        flip = np.random.randint(1,10000)%2  
        if flip:
            img = cv2.flip(img, 1)
        img = random_distort_image(img, hue, saturation, exposure)
        rgb_imgs.append(img)
        confs.append(max_iou)

    return rgb_imgs, confs




def worker(tag):
    rgb = cv2.resize(cv2.imread(os.path.join(object_dir, 'training',
                                             'image_2', tag + '.png')), (cfg.IMAGE_WIDTH, cfg.IMAGE_HEIGHT))
    lidar = np.fromfile(os.path.join(object_dir, 'training',
                                     'velodyne', tag + '.bin'), dtype=np.float32).reshape(-1, 4)
    label = np.array([line for line in open(os.path.join(
        object_dir, 'training', 'label_2', tag + '.txt'), 'r').readlines()])  # (N')
    cls = np.array([line.split()[0] for line in label])  # (N')
    gt_box3d = label_to_gt_box3d(np.array(label)[np.newaxis, :], cls='', coordinate='camera')[
        0]  # (N', 7) x, y, z, h, w, l, r

    choice = np.random.randint(1, 10)
    if choice >= 10:
        lidar_center_gt_box3d = camera_to_lidar_box(gt_box3d)
        lidar_corner_gt_box3d = center_to_corner_box3d(
            lidar_center_gt_box3d, coordinate='lidar')
        for idx in range(len(lidar_corner_gt_box3d)):
            # TODO: precisely gather the point
            t_rz = np.random.uniform(-np.pi / 10, np.pi / 10)
            t_x = np.random.normal()
            t_y = np.random.normal()
            t_z = np.random.normal()
            # check collision
            tmp = box_transform(
                lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')
            is_collision = False
            for idy in range(idx):
                x1, y1, w1, l1, r1 = tmp[0][[0, 1, 4, 5, 6]]
                x2, y2, w2, l2, r2 = lidar_center_gt_box3d[idy][[
                    0, 1, 4, 5, 6]]
                iou = cal_iou2d(np.array([x1, y1, w1, l1, r1], dtype=np.float32),
                                np.array([x2, y2, w2, l2, r2], dtype=np.float32))
                if iou > 0:
                    is_collision = True
                    break
            if not is_collision:
                box_corner = lidar_corner_gt_box3d[idx]
                minx = np.min(box_corner[:, 0])
                miny = np.min(box_corner[:, 1])
                minz = np.min(box_corner[:, 2])
                maxx = np.max(box_corner[:, 0])
                maxy = np.max(box_corner[:, 1])
                maxz = np.max(box_corner[:, 2])
                bound_x = np.logical_and(
                    lidar[:, 0] >= minx, lidar[:, 0] <= maxx)
                bound_y = np.logical_and(
                    lidar[:, 1] >= miny, lidar[:, 1] <= maxy)
                bound_z = np.logical_and(
                    lidar[:, 2] >= minz, lidar[:, 2] <= maxz)
                bound_box = np.logical_and(
                    np.logical_and(bound_x, bound_y), bound_z)
                lidar[bound_box, 0:3] = point_transform(
                    lidar[bound_box, 0:3], t_x, t_y, t_z, rz=t_rz)
                lidar_center_gt_box3d[idx] = box_transform(
                    lidar_center_gt_box3d[[idx]], t_x, t_y, t_z, t_rz, 'lidar')

        gt_box3d = lidar_to_camera_box(lidar_center_gt_box3d)
        newtag = 'aug_{}_1_{}'.format(
            tag, np.random.randint(1, args.aug_amount))
    elif True:
    # elif choice <= 7 and choice >= 2:
        # global rotation
        angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        lidar[:, 0:3] = point_transform(lidar[:, 0:3], 0, 0, 0, rz=angle)
        gt_box3d = box_transform(gt_box3d, 0, 0, 0, -angle, 'camera')
        newtag = 'aug_{}_2_{:.4f}'.format(tag, angle).replace('.', '_')
    else:
        # global scaling
        factor = np.random.uniform(0.95, 1.05)
        lidar = lidar * factor
        gt_box3d[:, 0:6] = gt_box3d[:, 0:6] * factor
        newtag = 'aug_{}_3_{:.4f}'.format(tag, factor).replace('.', '_')

    label = box3d_to_label(gt_box3d[np.newaxis, ...], cls[np.newaxis, ...], coordinate='camera')[0]  # (N')
    cv2.imwrite(os.path.join(output_path, 'image_2', newtag + '.png'), rgb)
    lidar.reshape(-1).tofile(os.path.join(output_path,
                                          'velodyne', newtag + '.bin'))
    voxel_dict = process_pointcloud(lidar)
    np.savez_compressed(os.path.join(
        output_path, 'voxel' if cfg.DETECT_OBJ == 'Car' else 'voxel_ped', newtag), **voxel_dict)
    with open(os.path.join(output_path, 'label_2', newtag + '.txt'), 'w+') as f:
        for line in label:
            f.write(line)
    print(newtag)


def main():
    fl = glob.glob(os.path.join(object_dir, 'training', 'calib', '*.txt'))
    candidate = [f.split('/')[-1].split('.')[0] for f in fl]
    tags = []
    for _ in range(args.aug_amount):
        tags.append(candidate[np.random.randint(0, len(candidate))])
    print('generate {} tags'.format(len(tags)))
    pool = mp.Pool(args.num_workers)
    pool.map(worker, tags)


if __name__ == '__main__':
    main()
