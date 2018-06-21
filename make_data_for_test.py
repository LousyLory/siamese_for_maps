# read words
import numpy as np
import cv2
import glob
from PIL import Image, ImageFont, ImageDraw
import os
import fnmatch
import matplotlib
import random
import math
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
import re
from tqdm import tqdm
import itertools
from scipy import ndimage
from operator import sub
from gen_small_sample_data import generate_word_images_from_list
from gen_small_sample_data import generate_left_words_from_image
import sys

map_dir = '/media/archan/maps_project/maps/'
anots_dir = '/media/archan/maps_project/annotations/current/'
list_of_maps = []
for i in glob.glob(map_dir+'*'):
	_,_,f = i.rpartition('/')
	f,_,_ = f.rpartition('.')
	list_of_maps.append(f)
print list_of_maps


fonts_list = []
for root, dirnames, filenames in os.walk('./fonts_new/'):
	for filename in fnmatch.filter(filenames, '*.ttf'):
        	fonts_list.append(os.path.join(root, filename))

background_images = []
for i in range(1, 6):
	my_file = Path('./map_textures/map_crop_0' + str(i) + '.jpg')
	if my_file.is_file():
		img = mpimg.imread('./map_textures/map_crop_0' + str(i) + '.jpg')
		background_images.append(img)


# file name leads
f1 = 'sythetic_word_images_nobg_nopad_'
f2 = 'list_of_words_nobg_nopad_'
f3 = 'original_words_nopad_'
f4 = 'original_images_nopad_'

for files in list_of_maps:
	list_of_words = []
	#map_image = cv2.imread(map_dir+files+'.tiff')
	dict_of_polygons = np.load(anots_dir+files+'.npy').item()

	for i in dict_of_polygons.keys():
		list_of_words.append(dict_of_polygons[i]['name'])
    
	sythetic_word_images = generate_word_images_from_list(list_of_words, fonts_list, background_images, padded=False, bg=False)
	np.save(f1+files+'.npy', sythetic_word_images)
	np.save(f2+files+'.npy', list_of_words)

for files in list_of_maps:
	original_words, _, original_images = generate_left_words_from_image(files, map_dir, anots_dir, padded=False)
	np.save(f3+files+'.npy', original_words)
	np.save(f4+files+'.npy', original_images)


