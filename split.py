#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import os
import glob
from misc_util import *
from misc_util import warn

def main():
	ratio = 0.85 # 0.7 for training, 0.3 for validation
	cur_dir = get_cur_dir()
	dataset_dir = os.path.join(cur_dir, 'data/object/training')
	# img_dir = os.path.join(dataset_dir, 'image_2')
	files = glob.glob(os.path.join(dataset_dir, 'image_2', '*.png'))
	files.sort()
	files = [file.split('/')[-1].split('.')[-2] for file in files]
	np.random.shuffle(files)
	warn("file: {}".format(files))
	warn("total : {}".format(len(files)))
	num_train = int(ratio * len(files))


	train_set = files[:num_train]
	valid_set = files[num_train:]

	warn("train: {}".format(len(train_set)))
	warn("valid: {}".format(len(valid_set)))

	nt = len(train_set)
	nv = len(valid_set)

	with open('trainset.txt', 'w+') as f:
		for idx in range(nt):
			f.write(train_set[idx] + '\n')
	f.close()
	with open('validset.txt', 'w+') as f:
		for idx in range(nv):
			f.write(valid_set[idx] + '\n')
	f.close()


	warn("total: {}".format(nt+nv))

if __name__ == '__main__':
    main()