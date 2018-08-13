# voxel_yolonet

Reproducing VoxelNet from Apple, forked from https://github.com/jeasinema/VoxelNet-tensorflow
Adding YOLO-structure

## SETTING 

### Preparing for final upload
Log: Aug 13, 2018, Monday

### Requirements may include:

- python 3.5
- tensorflow 1.4
- numpy
- opencv

### Folder structure
```
[root]
 data       : kitti dataset( : should be linked to your original kitti dataset)
 model      : model
 utils      : utils files
 
 and some files such as data_aug, evaluate_object, misc_utils, train .. are not located inside folders
 
```

### Structure of data folder

data -> object -> [training, testing, ...]
training -> [calib, image_2, label_2, velodyne, ..]
testing -> [calib, image_2, velodyne, ..] <- no label_2 folder in testing folder


## RUN

### train
All configurations are defined in config.py
```bash
cd [root]
CUDA_VISIBLE_DEVICES=2,3 python3.5 train.py
```
To use multiple gpus, you need to modify __C.GPU_AVAILABLE='0,1' in config.py file
If you use 3 gpus, __C.GPU_AVAILABLE='0,1,2' regardless of the actual numbering of your gpus such as (gpu 0, gpu 2, gpu 3) in CUDUA_VISIBLE_DEVICES=0,2,3

It will save model in 'save model' folder that will be automatically generated.
It also save images in 'save_image' folder which also be generated automatically.
And finally, it calls validation program to save the result in result folder and plot folder inside in it.

### test

Currently no test script.
It is included in validation step.
```
