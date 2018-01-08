
# -*- cooing:UTF-8 -*-

# File Name : utils.py
# Purpose :
# Creation Date : 09-12-2017
# Last Modified : Sat 23 Dec 2017 08:51:09 PM CST
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import cv2
import numpy as np
import shapely.geometry
import shapely.affinity
import math
from numba import jit

from config import cfg
from utils.box_overlaps import *
from misc_util import warn
import time

def lidar_to_bird_view(x, y, factor=1):
    a = (x - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor
    b = (y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor
    a = np.clip(a, a_max=(cfg.X_MAX - cfg.X_MIN) / cfg.VOXEL_X_SIZE * factor, a_min=0)
    b = np.clip(b, a_max=(cfg.Y_MAX - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor, a_min=0)
    return a, b


def angle_in_limit(angle):
    # To limit the angle in -pi/2 - pi/2
    limit_degree = 5
    while angle >= np.pi / 2:
        angle -= np.pi
    while angle < -np.pi / 2:
        angle += np.pi
    if abs(angle + np.pi / 2) < limit_degree / 180 * np.pi:
        angle = np.pi / 2
    return angle


def camera_to_lidar(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), p)
    p = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_T_VELO_2_CAM)), p)
    p = p[0:3]
    return tuple(p)


def lidar_to_camera(x, y, z):
    p = np.array([x, y, z, 1])
    p = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), p)
    p = np.matmul(np.array(cfg.MATRIX_R_RECT_0), p)
    p = p[0:3]
    return tuple(p)


def camera_to_lidar_point(points):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T  # (N,4) -> (4,N)

    points = np.matmul(np.linalg.inv(np.array(cfg.MATRIX_R_RECT_0)), points)
    points = np.matmul(np.linalg.inv(
        np.array(cfg.MATRIX_T_VELO_2_CAM)), points).T  # (4, N) -> (N, 4)
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def lidar_to_camera_point(points):
    # (N, 3) -> (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))]).T

    points = np.matmul(np.array(cfg.MATRIX_T_VELO_2_CAM), points)
    points = np.matmul(np.array(cfg.MATRIX_R_RECT_0), points).T
    points = points[:, 0:3]
    return points.reshape(-1, 3)


def camera_to_lidar_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, ry = box
        (x, y, z), h, w, l, rz = camera_to_lidar(
            x, y, z), h, w, l, -ry - np.pi / 2
        rz = angle_in_limit(rz)
        ret.append([x, y, z, h, w, l, rz])
    return np.array(ret).reshape(-1, 7)


def lidar_to_camera_box(boxes):
    # (N, 7) -> (N, 7) x,y,z,h,w,l,r
    ret = []
    for box in boxes:
        x, y, z, h, w, l, rz = box
        (x, y, z), h, w, l, ry = lidar_to_camera(
            x, y, z), h, w, l, -rz - np.pi / 2
        ry = angle_in_limit(ry)
        ret.append([x, y, z, h, w, l, ry])
    return np.array(ret).reshape(-1, 7)


def center_to_corner_box2d(boxes_center, coordinate='lidar', check = False):
    # (N, 5) -> (N, 4, 2)
    N = boxes_center.shape[0]
    boxes3d_center = np.zeros((N, 7))
    boxes3d_center[:, [0, 1, 4, 5, 6]] = boxes_center
    boxes3d_corner = center_to_corner_box3d(
        boxes3d_center, coordinate=coordinate)

    # if check == True:
    #     warn("center: {} corner: {}".format(boxes_center, boxes3d_corner))
    return boxes3d_corner[:, 0:4, 0:2]


def center_to_corner_box3d(boxes_center, coordinate='lidar'):
    # (N, 7) -> (N, 8, 3)
    N = boxes_center.shape[0]
    ret = np.zeros((N, 8, 3), dtype=np.float32)

    if coordinate == 'camera':
        boxes_center = camera_to_lidar_box(boxes_center)

    for i in range(N):
        box = boxes_center[i]
        translation = box[0:3]
        size = box[3:6]
        rotation = [0, 0, box[-1]]

        h, w, l = size[0], size[1], size[2]
        trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2], \
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], \
            [0, 0, 0, 0, h, h, h, h]])

        # re-create 3D bounding box in velodyne coordinate system
        yaw = rotation[2]
        rotMat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0]])
        cornerPosInVelo = np.dot(rotMat, trackletBox) + \
            np.tile(translation, (8, 1)).T
        box3d = cornerPosInVelo.transpose()
        ret[i] = box3d

    if coordinate == 'camera':
        for idx in range(len(ret)):
            ret[idx] = lidar_to_camera_point(ret[idx])

    return ret


def corner_to_center_box2d(boxes_corner, coordinate='lidar'):
    # (N, 4, 2) -> (N, 5)  x,y,w,l,r
    N = boxes_corner.shape[0]
    boxes3d_corner = np.zeros((N, 8, 3))
    boxes3d_corner[:, 0:4, 0:2] = boxes_corner
    boxes3d_corner[:, 4:8, 0:2] = boxes_corner
    boxes3d_center = corner_to_center_box3d(
        boxes3d_corner, coordinate=coordinate)

    return boxes3d_center[:, [0, 1, 4, 5, 6]]


def corner_to_standup_box2d(boxes_corner):
    # (N, 4, 2) -> (N, 4) x1, y1, x2, y2
    N = boxes_corner.shape[0]
    standup_boxes2d = np.zeros((N, 4))
    standup_boxes2d[:, 0] = np.min(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 1] = np.min(boxes_corner[:, :, 1], axis=1)
    standup_boxes2d[:, 2] = np.max(boxes_corner[:, :, 0], axis=1)
    standup_boxes2d[:, 3] = np.max(boxes_corner[:, :, 1], axis=1)

    return standup_boxes2d


# TODO: 0/90 may be not correct
def anchor_to_standup_box2d(anchors):
    # (N, 4) -> (N, 4) x,y,w,l -> x1,y1,x2,y2
    anchor_standup = np.zeros_like(anchors)
    # r == 0
    anchor_standup[::2, 0] = anchors[::2, 0] - anchors[::2, 3] / 2
    anchor_standup[::2, 1] = anchors[::2, 1] - anchors[::2, 2] / 2
    anchor_standup[::2, 2] = anchors[::2, 0] + anchors[::2, 3] / 2
    anchor_standup[::2, 3] = anchors[::2, 1] + anchors[::2, 2] / 2
    # r == pi/2
    anchor_standup[1::2, 0] = anchors[1::2, 0] - anchors[1::2, 2] / 2
    anchor_standup[1::2, 1] = anchors[1::2, 1] - anchors[1::2, 3] / 2
    anchor_standup[1::2, 2] = anchors[1::2, 0] + anchors[1::2, 2] / 2
    anchor_standup[1::2, 3] = anchors[1::2, 1] + anchors[1::2, 3] / 2

    # warn(" in : {}".format(anchors))
    # warn(" out: {}".format(anchor_standup))

    return anchor_standup


def corner_to_center_box3d(boxes_corner, coordinate='camera'):
    # (N, 8, 3) -> (N, 7) x,y,z,h,w,l,ry/z
    if coordinate == 'lidar':
        for idx in range(len(boxes_corner)):
            boxes_corner[idx] = lidar_to_camera_point(boxes_corner[idx])
    ret = []
    for roi in boxes_corner:
        if cfg.CORNER2CENTER_AVG:  # average version
            roi = np.array(roi)
            h = abs(np.sum(roi[:4, 1] - roi[4:, 1]) / 4)
            w = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            ) / 4
            l = np.sum(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            ) / 4
            x, y, z = np.sum(roi, axis=0) / 8
            y = y + h/2
            # warn("x {} y {} z {}".format(x, y, z))


            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        else:  # max version
            h = max(abs(roi[:4, 1] - roi[4:, 1]))
            w = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[1, [0, 2]] - roi[2, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[7, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[5, [0, 2]] - roi[6, [0, 2]])**2))
            )
            l = np.max(
                np.sqrt(np.sum((roi[0, [0, 2]] - roi[1, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[2, [0, 2]] - roi[3, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[4, [0, 2]] - roi[5, [0, 2]])**2)) +
                np.sqrt(np.sum((roi[6, [0, 2]] - roi[7, [0, 2]])**2))
            )
            x, y, z = np.sum(roi, axis=0) / 8
            y = y + h/2
            ry = np.sum(
                math.atan2(roi[2, 0] - roi[1, 0], roi[2, 2] - roi[1, 2]) +
                math.atan2(roi[6, 0] - roi[5, 0], roi[6, 2] - roi[5, 2]) +
                math.atan2(roi[3, 0] - roi[0, 0], roi[3, 2] - roi[0, 2]) +
                math.atan2(roi[7, 0] - roi[4, 0], roi[7, 2] - roi[4, 2]) +
                math.atan2(roi[0, 2] - roi[1, 2], roi[1, 0] - roi[0, 0]) +
                math.atan2(roi[4, 2] - roi[5, 2], roi[5, 0] - roi[4, 0]) +
                math.atan2(roi[3, 2] - roi[2, 2], roi[2, 0] - roi[3, 0]) +
                math.atan2(roi[7, 2] - roi[6, 2], roi[6, 0] - roi[7, 0])
            ) / 8
            if w > l:
                w, l = l, w
                ry = angle_in_limit(ry + np.pi / 2)
        ret.append([x, y, z, h, w, l, ry])
        # warn("z: {}".format(x))
    if coordinate == 'lidar':
        ret = camera_to_lidar_box(np.array(ret))

    return np.array(ret)


# this just for visulize and testing
def lidar_box3d_to_camera_box(boxes3d, cal_projection=False):
    # (N, 7) -> (N, 4)/(N, 8, 2)  x,y,z,h,w,l,rz -> x1,y1,x2,y2/8*(x, y)
    num = len(boxes3d)
    boxes2d = np.zeros((num, 4), dtype=np.int32)
    projections = np.zeros((num, 8, 2), dtype=np.float32)

    lidar_boxes3d_corner = center_to_corner_box3d(boxes3d, coordinate='lidar')
    P2 = np.array(cfg.MATRIX_P2)

    for n in range(num):
        box3d = lidar_boxes3d_corner[n]
        box3d = lidar_to_camera_point(box3d)
        points = np.hstack((box3d, np.ones((8, 1)))).T  # (8, 4) -> (4, 8)
        points = np.matmul(P2, points).T
        points[:, 0] /= points[:, 2]
        points[:, 1] /= points[:, 2]

        projections[n] = points[:, 0:2]
        minx = int(np.min(points[:, 0]))
        maxx = int(np.max(points[:, 0]))
        miny = int(np.min(points[:, 1]))
        maxy = int(np.max(points[:, 1]))

        boxes2d[n, :] = minx, miny, maxx, maxy

    return projections if cal_projection else boxes2d


def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (cfg.INPUT_HEIGHT * factor, cfg.INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if cfg.X_MIN < x < cfg.X_MAX and cfg.Y_MIN < y < cfg.Y_MAX:
            x, y = int((x - cfg.X_MIN) / cfg.VOXEL_X_SIZE *
                       factor), int((y - cfg.Y_MIN) / cfg.VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


def draw_lidar_box3d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]),
                              color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=True)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

    # draw gt projections
    for qs in gt_projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), gt_color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def draw_lidar_box3d_on_image_test(img, boxes3d, scores, color=(0, 255, 255), thickness=1):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=True)

    # draw projections
    for qs in projections:
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

            i, j = k, k + 4
            cv2.line(img, (qs[i, 0], qs[i, 1]), (qs[j, 0],
                                                 qs[j, 1]), color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def draw_lidar_box3d_to_bbox2d_on_image(img, boxes3d, scores, gt_boxes3d=np.array([]), color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1):
    # Input:
    #   img: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = img.copy()
    projections = lidar_box3d_to_camera_box(boxes3d, cal_projection=False)
    gt_projections = lidar_box3d_to_camera_box(gt_boxes3d, cal_projection=False)

    # draw projections
    for pr in projections:
        x1 = pr[0]
        y1 = pr[1]
        x2 = pr[2]
        y2 = pr[3]
        cv2.line(img, (x1, y1), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y1), color, thickness, cv2.LINE_AA)

    for gt in gt_projections:
        x1 = gt[0]
        y1 = gt[1]
        x2 = gt[2]
        y2 = gt[3]
        cv2.line(img, (x1, y1), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y1), color, thickness, cv2.LINE_AA)


    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def draw_bbox2d_on_image(img, boxes2d, color=(0,255,255), thickness=1):
    img = img.copy()

    for box in boxes2d:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        cv2.line(img, (x1, y1), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x2, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x1, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y1), color, thickness, cv2.LINE_AA)
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)



def draw_lidar_box3d_on_birdview(birdview, boxes3d, scores, gt_boxes3d=np.array([]),
                                 color=(0, 255, 255), gt_color=(255, 0, 255), thickness=1, factor=1):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar')
    corner_gt_boxes3d = center_to_corner_box3d(gt_boxes3d, coordinate='lidar')
    # draw gt
    for box in corner_gt_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 gt_color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 gt_color, thickness, cv2.LINE_AA)

    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        # warn("x0: {} y0: {} x1: {} y1: {} x2: {} y2: {} x3: {} y3:{}".format(int(x0), int(y0), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)))

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

def draw_lidar_box3d_on_birdview_test(birdview, boxes3d, scores, color=(0, 255, 255), thickness=1, factor=1):
    # Input:
    #   birdview: (h, w, 3)
    #   boxes3d (N, 7) [x, y, z, h, w, l, r]
    #   scores
    #   gt_boxes3d (N, 7) [x, y, z, h, w, l, r]
    img = birdview.copy()
    corner_boxes3d = center_to_corner_box3d(boxes3d, coordinate='lidar')
    # draw detections
    for box in corner_boxes3d:
        x0, y0 = lidar_to_bird_view(*box[0, 0:2], factor=factor)
        x1, y1 = lidar_to_bird_view(*box[1, 0:2], factor=factor)
        x2, y2 = lidar_to_bird_view(*box[2, 0:2], factor=factor)
        x3, y3 = lidar_to_bird_view(*box[3, 0:2], factor=factor)

        # warn("x0: {} y0: {} x1: {} y1: {} x2: {} y2: {} x3: {} y3:{}".format(int(x0), int(y0), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)))

        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x2), int(y2)), (int(x3), int(y3)),
                 color, thickness, cv2.LINE_AA)
        cv2.line(img, (int(x3), int(y3)), (int(x0), int(y0)),
                 color, thickness, cv2.LINE_AA)

    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)


def label_to_gt_box3d(labels, cls='Car', coordinate='camera'):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes3d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else:
        acc_cls = []

    for label in labels:
        boxes3d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                h, w, l, x, y, z, r = [float(i) for i in ret[-7:]]
                box3d = np.array([x, y, z, h, w, l, r])
                boxes3d_a_label.append(box3d)
        if coordinate == 'lidar':
            boxes3d_a_label = camera_to_lidar_box(np.array(boxes3d_a_label))

        boxes3d.append(np.array(boxes3d_a_label).reshape(-1, 7))
    return boxes3d

def label_to_gt_box2d(labels, cls='Car', coordinate='camera'):
    # Input:
    #   label: (N, N')
    #   cls: 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate: 'camera' or 'lidar'
    # Output:
    #   (N, N', 7)
    boxes2d = []
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else:
        acc_cls = []

    for label in labels:
        boxes2d_a_label = []
        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                x_min, y_min, x_max, y_max = [float(i) for i in ret[4:8]]
                box2d = np.array([x_min, y_min, x_max, y_max])
                boxes2d_a_label.append(box2d)

        boxes2d.append(np.array(boxes2d_a_label).reshape(-1, 4))
    return boxes2d

def label_to_num_obj(f_labels, cls='Car'):
    if cls == 'Car':
        acc_cls = ['Car', 'Van']
    elif cls == 'Pedestrian':
        acc_cls = ['Pedestrian']
    elif cls == 'Cyclist':
        acc_cls = ['Cyclist']
    else:
        acc_cls = []

    num_obj = 0 

    num_f_labels = len(f_labels)

    for idx, f_label in enumerate(f_labels):
        # if idx % 10 == 0:
        #     warn("loading {} / {}".format(idx, num_f_labels))
        label = np.array([line for line in open(f_label, 'r').readlines()])

        for line in label:
            ret = line.split()
            if ret[0] in acc_cls or acc_cls == []:
                num_obj = num_obj + 1
    return num_obj

def box3d_to_label(batch_box3d, batch_cls, batch_score=[], include_score = False, coordinate='camera'):
    # Input:
    #   (N, N', 7) x y z h w l r
    #   (N, N')
    #   cls: (N, N') 'Car' or 'Pedestrain' or 'Cyclist'
    #   coordinate(input): 'camera' or 'lidar'
    # Output:
    #   label: (N, N') N batches and N lines
    # warn("to label")
    batch_label = []
    # warn("shape: {} {} {}".format(np.shape(batch_box3d), np.shape(batch_cls), np.shape(batch_score)))
    if include_score:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(15)]) + '\n'
        for boxes, scores, clses in zip(batch_box3d, batch_score, batch_cls):
            label = []
            for box, score, cls in zip(boxes, scores, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32)), cal_projection=False)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32))[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(
                    cls, 0, 0, -10, *box2d, *box3d, float(score)))
            batch_label.append(label)
    else:
        template = '{} ' + ' '.join(['{:.4f}' for i in range(14)]) + '\n'
        for boxes, clses in zip(batch_box3d, batch_cls):
            label = []
            for box, cls in zip(boxes, clses):
                if coordinate == 'camera':
                    box3d = box
                    box2d = lidar_box3d_to_camera_box(
                        camera_to_lidar_box(box[np.newaxis, :].astype(np.float32)), cal_projection=False)[0]
                else:
                    box3d = lidar_to_camera_box(
                        box[np.newaxis, :].astype(np.float32))[0]
                    box2d = lidar_box3d_to_camera_box(
                        box[np.newaxis, :].astype(np.float32), cal_projection=False)[0]
                x, y, z, h, w, l, r = box3d
                box3d = [h, w, l, x, y, z, r]
                label.append(template.format(cls, 0, 0, 0, *box2d, *box3d))
            batch_label.append(label)
        # warn("batch_label: {}".format(batch_label))

    return batch_label


def bbox_iou(box1, box2, x1y1x2y2=True):
    # warn("box1: {}, box2: {}".format(box1, box2))
    if x1y1x2y2:
        mx = min(box1[0], box2[0])
        Mx = max(box1[2], box2[2])
        my = min(box1[1], box2[1])
        My = max(box1[3], box2[3])
        w1 = box1[2] - box1[0]
        h1 = box1[3] - box1[1]
        w2 = box2[2] - box2[0]
        h2 = box2[3] - box2[1]
    else:
        mx = min(box1[0]-box1[2]/2.0, box2[0]-box2[2]/2.0)
        Mx = max(box1[0]+box1[2]/2.0, box2[0]+box2[2]/2.0)
        my = min(box1[1]-box1[3]/2.0, box2[1]-box2[3]/2.0)
        My = max(box1[1]+box1[3]/2.0, box2[1]+box2[3]/2.0)
        w1 = box1[2]
        h1 = box1[3]
        w2 = box2[2]
        h2 = box2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    carea = 0
    if cw <= 0 or ch <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    uarea = area1 + area2 - carea
    return carea/uarea



def cal_anchors():
    # Output:
    #   anchors: (w, l, 2, 7) x y z h w l r
    x = np.linspace(cfg.X_MIN, cfg.X_MAX, cfg.FEATURE_WIDTH)
    y = np.linspace(cfg.Y_MIN, cfg.Y_MAX, cfg.FEATURE_HEIGHT)
    cx, cy = np.meshgrid(x, y)
    # all is (w, l, 2)
    cx = np.tile(cx[..., np.newaxis], 2)
    cy = np.tile(cy[..., np.newaxis], 2)
    cz = np.ones_like(cx) * cfg.ANCHOR_Z
    w = np.ones_like(cx) * cfg.ANCHOR_W
    l = np.ones_like(cx) * cfg.ANCHOR_L
    h = np.ones_like(cx) * cfg.ANCHOR_H
    r = np.ones_like(cx)
    r[..., 0] = 0  # 0
    r[..., 1] = 90 / 180 * np.pi  # 90

    # 7*(w,l,2) -> (w, l, 2, 7)
    anchors = np.stack([cx, cy, cz, h, w, l, r], axis=-1)

    return anchors

def cal_rpn_target(labels, feature_map_shape, anchors, cls='Car', coordinate='lidar'):
    # Input:
    #   labels: (N, N')
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)
    # Output:
    #   pos_equal_one (N, w, l, 2)
    #   neg_equal_one (N, w, l, 2)
    #   targets (N, w, l, 14)
    # attention: cal IoU on birdview
    batch_size = labels.shape[0]
    batch_gt_boxes3d = label_to_gt_box3d(labels, cls=cls, coordinate=coordinate)
    # defined in eq(1) in 2.2
    anchors_reshaped = anchors.reshape(-1, 7)
    # anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    pos_equal_one = np.zeros((batch_size, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2))
    neg_equal_one = np.ones((batch_size, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 2))
    targets = np.zeros((batch_size, cfg.FEATURE_HEIGHT, cfg.FEATURE_WIDTH, 14))

    anchor_origin = np.array([[0, 0, cfg.ANCHOR_W, cfg.ANCHOR_L],[0, 0, cfg.ANCHOR_W, cfg.ANCHOR_L]])
    anchor_standup_2d_origin = anchor_to_standup_box2d(anchor_origin)
    anchor_rot = [0, 90 / 180 * np.pi]
    anchor_d = np.sqrt(cfg.ANCHOR_W**2 + cfg.ANCHOR_L**2)
    # warn("shape: {}".format(batch_gt_boxes3d[0][0, [0, 1, 4, 5, 6]]))
    for batch_id in range(batch_size):
        t0 = time.time()
        for t in range(len(batch_gt_boxes3d[batch_id])):
            gx, gy, gz, gh, gw, gl, gr = batch_gt_boxes3d[batch_id][t, [0, 1, 2, 3, 4, 5, 6]]

            if (gx > cfg.X_MAX) or (gx < cfg.X_MIN) or (gy > cfg.Y_MAX) or (gy < cfg.Y_MIN):
                warn("***  illegal data removed: {:.2f} {:.2f} ***".format(gx, gy))
                continue

            gx_ratio = (gx - cfg.X_MIN) / (cfg.X_MAX - cfg.X_MIN) * (cfg.FEATURE_WIDTH-1)
            gy_ratio = (gy - cfg.Y_MIN) / (cfg.Y_MAX - cfg.Y_MIN) * (cfg.FEATURE_HEIGHT-1)
            gi = int(gx_ratio)
            gj = int(gy_ratio)

            gt_box_origin = np.array([[0, 0, gw, gl, gr]])
            gt_standup_2d_origin = corner_to_standup_box2d(center_to_corner_box2d(gt_box_origin, coordinate=coordinate))
            # warn("anchor: {}".format(gt_standup_2d_origin))
            best_iou = 0
            best_anchor = 0
            for anchor in range(len(anchor_origin)):
                iou = bbox_iou(anchor_standup_2d_origin[anchor], gt_standup_2d_origin[0], x1y1x2y2 = True)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = anchor 
                # warn("{} : iou : {}".format(anchor, iou))
            # warn("shape iou: {}".format(np.shape(iou)))
            # best_anchor_id = np.argmax(iou.T, axis=1)
            
            index_x = gi
            index_y = gj
            index_z = best_anchor

            # warn("{} : gx {:.2f} gy {:.2f} gx_ratio {:.2f} gy_ratio {:.2f} gw {:.2f} gl {:.2f} [ gi {} gj {} anchor {} ] iou {:.2f}".format(t, gx, gy, gx_ratio, gy_ratio, gw, gl, gi, gj, best_anchor, best_iou))

            pos_equal_one[batch_id, index_y, index_x, best_anchor] = 1
            neg_equal_one[batch_id, index_y, index_x, best_anchor] = 0

            targets[batch_id, index_y, index_x, np.array(index_z) * 7] = gx_ratio - gi 
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 1] = gy_ratio - gj
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 2] = (gz - cfg.ANCHOR_Z) / cfg.ANCHOR_H
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 3] = np.log(gh / cfg.ANCHOR_H)
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 4] = np.log(gw / cfg.ANCHOR_W)
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 5] = np.log(gl / cfg.ANCHOR_L)
            targets[batch_id, index_y, index_x, np.array(index_z) * 7 + 6] = (gr - anchor_rot[best_anchor])
        # t1 = time.time()
        # # warn("time for rpn : {}".format(t1-t0))
        # # warn("feature map :{} ".format(np.shape(targets)))

        # # BOTTLENECK
        # anchors_standup_2d = anchor_to_standup_box2d(
        #     anchors_reshaped[:, [0, 1, 4, 5]])
        # # warn("anchor gt: {}".format(anchors_standup_2d[0:4]))
        # # BOTTLENECK
        # gt_standup_2d = corner_to_standup_box2d(center_to_corner_box2d(
        #     batch_gt_boxes3d[batch_id][:, [0, 1, 4, 5, 6]], coordinate=coordinate))

        # iou = bbox_overlaps(
        #     np.ascontiguousarray(anchors_standup_2d).astype(np.float32),
        #     np.ascontiguousarray(gt_standup_2d).astype(np.float32),
        # )

        # # find anchor with highest iou(iou should also > 0)
        # id_highest = np.argmax(iou.T, axis=1)
        # id_highest_gt = np.arange(iou.T.shape[0])
        # mask = iou.T[id_highest_gt, id_highest] > 0
        # id_highest, id_highest_gt = id_highest[mask], id_highest_gt[mask]

        # # find anchor iou > cfg.XXX_POS_IOU
        # id_pos, id_pos_gt = np.where(iou > cfg.RPN_POS_IOU)

        # # find anchor iou < cfg.XXX_NEG_IOU
        # id_neg = np.where(np.sum(iou < cfg.RPN_NEG_IOU,
        #                          axis=1) == iou.shape[1])[0]

        # id_pos = np.concatenate([id_pos, id_highest])
        # id_pos_gt = np.concatenate([id_pos_gt, id_highest_gt])

        # # TODO: uniquify the array in a more scientific way
        # id_pos, index = np.unique(id_pos, return_index=True)
        # id_pos_gt = id_pos_gt[index]
        # id_neg.sort()

        # # cal the target and set the equal one
        # index_x, index_y, index_z = np.unravel_index(
        #     id_pos, (*feature_map_shape, 2))
        # pos_equal_one[batch_id, index_x, index_y, index_z] = 1

        # for k in range(len(index_x)):
        #     warn("x {} y {} z {}".format(index_x[k], index_y[k], index_z[k]))
        # # warn("x: {}".format(index_x))
        # # warn("y: {}".format(index_y))

        # # ATTENTION: index_z should be np.array
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7] = (
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 0] - anchors_reshaped[id_pos, 0]) / anchors_d[id_pos]
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 1] = (
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 1] - anchors_reshaped[id_pos, 1]) / anchors_d[id_pos]
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 2] = (
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 2] - anchors_reshaped[id_pos, 2]) / cfg.ANCHOR_H
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 3] = np.log(
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 3] / anchors_reshaped[id_pos, 3])
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 4] = np.log(
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 4] / anchors_reshaped[id_pos, 4])
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 5] = np.log(
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 5] / anchors_reshaped[id_pos, 5])
        # targets[batch_id, index_x, index_y, np.array(index_z) * 7 + 6] = (
        #     batch_gt_boxes3d[batch_id][id_pos_gt, 6] - anchors_reshaped[id_pos, 6])

        # index_x, index_y, index_z = np.unravel_index(
        #     id_neg, (*feature_map_shape, 2))
        # neg_equal_one[batch_id, index_x, index_y, index_z] = 1
        # # to avoid a box be pos/neg in the same time
        # index_x, index_y, index_z = np.unravel_index(
        #     id_highest, (*feature_map_shape, 2))
        # neg_equal_one[batch_id, index_x, index_y, index_z] = 0

    return pos_equal_one, neg_equal_one, targets


# BOTTLENECK
def delta_to_boxes3d(deltas, anchors, coordinate='lidar'):
    # Input:
    #   deltas: (N, w, l, 14)
    #   feature_map_shape: (w, l)
    #   anchors: (w, l, 2, 7)

    # Ouput:
    #   boxes3d: (N, w*l*2, 7)
    anchors_reshaped = anchors.reshape(-1, 7)
    deltas = deltas.reshape(deltas.shape[0], -1, 7)
    anchors_d = np.sqrt(anchors_reshaped[:, 4]**2 + anchors_reshaped[:, 5]**2)
    boxes3d = np.zeros_like(deltas)
    # warn("check deltas: {}".format(deltas[..., [0]]))
    boxes3d[..., [0]] = deltas[..., [0]] * cfg.FEATURE_WIDTH_ACTUAL + anchors_reshaped[..., [0]]
    boxes3d[..., [1]] = deltas[..., [1]] * cfg.FEATURE_HEIGHT_ACTUAL + anchors_reshaped[..., [1]]
    boxes3d[..., [2]] = deltas[..., [2]] * cfg.ANCHOR_H + anchors_reshaped[..., [2]]
    boxes3d[..., [3, 4, 5]] = np.exp(
        deltas[..., [3, 4, 5]]) * anchors_reshaped[..., [3, 4, 5]]
    boxes3d[..., 6] = deltas[..., 6] + anchors_reshaped[..., 6]
    # warn("check calu: {}".format(boxes3d[..., [0]]))

    return boxes3d


def point_transform(points, tx, ty, tz, rx=0, ry=0, rz=0):
    # Input:
    #   points: (N, 3)
    #   rx/y/z: in radians
    # Output:
    #   points: (N, 3)
    N = points.shape[0]
    points = np.hstack([points, np.ones((N, 1))])

    mat1 = np.eye(4)
    mat1[3, 0:3] = tx, ty, tz
    points = np.matmul(points, mat1)

    if rx != 0:
        mat = np.zeros((4, 4))
        mat[0, 0] = 1
        mat[3, 3] = 1
        mat[1, 1] = np.cos(rx)
        mat[1, 2] = -np.sin(rx)
        mat[2, 1] = np.sin(rx)
        mat[2, 2] = np.cos(rx)
        points = np.matmul(points, mat)

    if ry != 0:
        mat = np.zeros((4, 4))
        mat[1, 1] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(ry)
        mat[0, 2] = np.sin(ry)
        mat[2, 0] = -np.sin(ry)
        mat[2, 2] = np.cos(ry)
        points = np.matmul(points, mat)

    if rz != 0:
        mat = np.zeros((4, 4))
        mat[2, 2] = 1
        mat[3, 3] = 1
        mat[0, 0] = np.cos(rz)
        mat[0, 1] = -np.sin(rz)
        mat[1, 0] = np.sin(rz)
        mat[1, 1] = np.cos(rz)
        points = np.matmul(points, mat)

    return points[:, 0:3]


def box_transform(boxes, tx, ty, tz, r=0, coordinate='lidar'):
    # Input:
    #   boxes: (N, 7) x y z h w l rz/y
    # Output:
    #   boxes: (N, 7) x y z h w l rz/y
    boxes_corner = center_to_corner_box3d(
        boxes, coordinate=coordinate)  # (N, 8, 3)
    for idx in range(len(boxes_corner)):
        if coordinate == 'lidar':
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, rz=r)
        else:
            boxes_corner[idx] = point_transform(
                boxes_corner[idx], tx, ty, tz, ry=r)


    return corner_to_center_box3d(boxes_corner, coordinate=coordinate)


def cal_iou2d(box1, box2):
    # Input:
    #   box1/2: x, y, w, l, r
    # Output:
    #   iou
    x1, y1, w1, l1, r1 = box1
    x2, y2, w2, l2, r2 = box2
    c1 = shapely.geometry.box(-w1 / 2.0, -l1 / 2.0, w1 / 2.0, l1 / 2.0)
    c2 = shapely.geometry.box(-w2 / 2.0, -l2 / 2.0, w2 / 2.0, l2 / 2.0)

    c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
    c2 = shapely.affinity.rotate(c2, r2, use_radians=True)

    c1 = shapely.affinity.translate(c1, x1, y1)
    c2 = shapely.affinity.translate(c2, x2, y2)

    intersect = c1.intersection(c2)

    return intersect.area / (c1.area + c2.area - intersect.area)


def cal_z_intersect(cz1, h1, cz2, h2):
    b1z1, b1z2 = cz1 - h1 / 2, cz1 + h1 / 2
    b2z1, b2z2 = cz2 - h2 / 2, cz2 + h2 / 2
    if b1z1 > b2z2 or b2z1 > b1z2:
        return 0
    elif b2z1 <= b1z1 <= b2z2:
        if b1z2 <= b2z2:
            return h1 / h2
        else:
            return (b2z2 - b1z1) / (b1z2 - b2z1)
    elif b1z1 < b2z1 < b1z2:
        if b2z2 <= b1z2:
            return h2 / h1
        else:
            return (b1z2 - b2z1) / (b2z2 - b1z1)


def cal_iou3d(box1, box2):
    # Input:
    #   box1/2: x, y, z, h, w, l, r
    # Output:
    #   iou

    x1, y1, z1, h1, w1, l1, r1 = box1
    x2, y2, z2, h2, w2, l2, r2 = box2
    c1 = shapely.geometry.box(-w1 / 2.0, -l1 / 2.0, w1 / 2.0, l1 / 2.0)
    c2 = shapely.geometry.box(-w2 / 2.0, -l2 / 2.0, w2 / 2.0, l2 / 2.0)

    c1 = shapely.affinity.rotate(c1, r1, use_radians=True)
    c2 = shapely.affinity.rotate(c2, r2, use_radians=True)

    c1 = shapely.affinity.translate(c1, x1, y1)
    c2 = shapely.affinity.translate(c2, x2, y2)

    z_intersect = cal_z_intersect(z1, h1, z2, h2)

    intersect = c1.intersection(c2)

    return intersect.area * z_intersect / (c1.area * h1 + c2.area * h2 - intersect.area * z_intersect)


def cal_box3d_iou(boxes3d, gt_boxes3d, cal_3d=0):
    # Inputs:
    #   boxes3d: (N1, 7) x,y,z,h,w,l,r
    #   gt_boxed3d: (N2, 7) x,y,z,h,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes3d)
    N2 = len(gt_boxes3d)
    output = np.zeros((N1, N2), dtype=np.float32)

    for idx in range(N1):
        for idy in range(N2):
            if cal_3d:
                output[idx, idy] = float(
                    cal_iou3d(boxes3d[idx], gt_boxes3d[idy]))
            else:
                output[idx, idy] = float(
                    cal_iou2d(boxes3d[idx, [0, 1, 4, 5, 6]], gt_boxes3d[idy, [0, 1, 4, 5, 6]]))

    return output


def cal_box2d_iou(boxes2d, gt_boxes2d):
    # Inputs:
    #   boxes2d: (N1, 5) x,y,w,l,r
    #   gt_boxes2d: (N2, 5) x,y,w,l,r
    # Outputs:
    #   iou: (N1, N2)
    N1 = len(boxes2d)
    N2 = len(gt_boxes2d)
    output = np.zeros((N1, N2), dtype=np.float32)
    for idx in range(N1):
        for idy in range(N2):
            output[idx, idy] = cal_iou2d(boxes2d[idx], gt_boxes2d[idy])

    return output

def load_kitti_calib(velo_calib_path):


    with open(velo_calib_path) as fi:
        lines = fi.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)
        
    return {'P2' : P2.reshape(3,4),
            'P3' : P3.reshape(3,4),
            'R0' : R0.reshape(3,3),
            'Tr_velo2cam' : Tr_velo_to_cam.reshape(3, 4)}

def calib_gathered(raw_calib):
    calib = np.zeros((4, 12))
    calib[0, :] = raw_calib['P2'].reshape(12)
    calib[1, :] = raw_calib['P3'].reshape(12)
    calib[2, :9] = raw_calib['R0'].reshape(9)
    calib[3, :] = raw_calib['Tr_velo2cam'].reshape(12)

    return calib

def calib_to_P(calib):
    #WZN: get the actual overall calibration matrix from Lidar coord to image
    #calib is 4*12 read by imdb
    #P is 3*4 s.t. uvw=P*XYZ1
    C2V = np.vstack((np.reshape(calib[3,:],(3,4)),np.array([0,0,0,1])))
    R0 = np.hstack((np.reshape(calib[2,:],(4,3)),np.array([[0],[0],[0],[1]])))
    P2 = np.reshape(calib[0,:],(3,4))
    P = np.matmul(np.matmul(P2,R0),C2V)
    return P

def projectToImage(pts_3D, P):
    """
    PROJECTTOIMAGE projects 3D points in given coordinate system in the image
    plane using the given projection matrix P.

    Usage: pts_2D = projectToImage(pts_3D, P)
    input: pts_3D: 3xn matrix
          P:      3x4 projection matrix
    output: pts_2D: 2xn matrix

    last edited on: 2012-02-27
    Philip Lenz - lenz@kit.edu
    """
    # project in image
    mat = np.vstack((pts_3D, np.ones((pts_3D.shape[1]))))

    pts_2D = np.dot(P, mat)

    # scale projected points
    pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
    pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]
    pts_2D = np.delete(pts_2D, 2, 0)

    return pts_2D

def distort_image(img, hue, sat, val):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # warn("hsv_img shape: {}".format(np.shape(hsv_img)))
    
    h, s, v = cv2.split(hsv_img)

    h = h + hue*255
    h[h>255] -= 255
    h[h<0] += 255

    s = s * sat
    v = v * val

    h = h.astype(np.uint8)
    s = s.astype(np.uint8)
    v = v.astype(np.uint8)

    hsv_img = cv2.merge([h,s,v])
    out = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

    return img

def rand_scale(s):
    scale = np.random.uniform(1, s)
    if(np.random.randint(1,10000)%2): 
        return scale
    return 1./scale

def random_distort_image(img, hue, saturation, exposure):
    dhue = np.random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(img, dhue, dsat, dexp)
    return res



if __name__ == '__main__':
    pass
