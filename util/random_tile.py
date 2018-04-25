import random
import cv2
import numpy as np
import os

def tile_function(tile_dir, input_img_path, img_number=412, tile_num=4, tile_times=20):
	tile = tile_num
	if not os.path.isdir(tile_dir):
		os.mkdir(tile_dir)
	for num_time in range(tile_times):
		out_img_path = os.path.join(tile_dir, "{}_{}_tile.jpg".format(img_number, num_time))
		input_img = cv2.imread(input_img_path)
		height,width = input_img.shape[0:2]

		width_step = int(width / tile)
		height_step = int(height / tile)

		tile_coors = []

		for i in range(tile):
			for j in range(tile):
				tile_coors.append((i * width_step, j * height_step))

		random.shuffle(tile_coors)

		coor_count = 0
		out_img = np.zeros(input_img.shape)
		print(out_img.shape)

		for i in range(tile):
			for j in range(tile):
				out_img[ j * height_step:(j + 1) * height_step, i * width_step:(i + 1)* width_step, :] = input_img[tile_coors[coor_count][1]:tile_coors[coor_count][1]+height_step, tile_coors[coor_count][0]:tile_coors[coor_count][0]+width_step, :]
				coor_count = coor_count + 1

		cv2.imwrite(out_img_path, out_img)

if __name__ == '__main__':
	tile_function('.datasets/half/3/tile', '.datasets/half/3/train/3.jpg', img_number=239, tile_num=4, tile_times=20)
