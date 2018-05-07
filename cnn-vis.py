# some code from: https://github.com/thesemicolonguy/convisualize_nb
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
# %matplotlib inline
from PIL import Image
import json
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

# from skimage import io, exposure, color

def entropy(band, bins=10):
    hist, _ = np.histogram(band, bins=10)
    hist = hist[hist > 0]
    return -np.log2(hist / float(hist.sum())).sum()

def filter_outputs(image, layer_to_visualize, modulelist, save_dir):
    output = None
    name = None

    length = len(modulelist)

    for count in range(length):
        layer = modulelist[str(count)]
        image = layer(image)
        if count == layer_to_visualize:
            output = image
            name = str(layer)

    filters = []
    output = output.data.squeeze()

    for i in range(output.shape[0]):
        filters.append(output[i,:,:])

    save_filters = []
    filters_entropy = {}
    for filter in filters:
        en = entropy(filter)
        filters_entropy[filter] = en
    filters_entropy = sorted(filters_entropy.items(), key=lambda item:item[1])
    # print(filters_entropy)
    filters = filters[:20]

    # fig = plt.figure()
    width, height = int(filters[0].size()[0]), int(filters[0].size()[1])
    times = width / float(height)
    plt.rcParams["figure.figsize"] = (1, times)

    for i in range(len(filters)):
        plt.axis('off')
        plt.imshow(filters[i], cmap='jet', interpolation='bilinear')
        plt.savefig('{}/{}_{:.4}_{}.png'.format(save_dir, layer_to_visualize, name, i), dpi=2 * width)

def visulize_all_channel_into_one(image, layer_to_visualize, modulelist, save_dir):
    output = None
    name = None

    length = len(modulelist)

    for count in range(length):
        layer = modulelist[str(count)]
        image = layer(image)
        if count == layer_to_visualize:
            output = image
            name = str(layer)

    output = output.data.squeeze()
    output = output.cpu().numpy()
    #for i in range(output.shape[0]):
    #    max_interval, min_interval = np.max(output[i, :, :]), np.min(output[i, :, :])
    #    output[i, :, :] = output[i, :, :] / (max_interval - min_interval) + 0.5
    # output = output * output
    output = np.mean(output, axis=0)

    # fig = plt.figure()
    height, width = int(output.shape[0]), int(output.shape[1])
    times = height / float(width)
    plt.rcParams["figure.figsize"] = (1, times)

    # for i in range(len(filters)):
    plt.axis('off')
    plt.imshow(output, cmap='jet', interpolation='bilinear')
    #plt.savefig('{}/{}_{:.4}.png'.format(save_dir, layer_to_visualize, name), dpi=2 * height, transparent = True, bbox_inches = 'tight', pad_inches = 0)
    plt.savefig('{}/{}_{:.4}.png'.format(save_dir, layer_to_visualize, name), dpi=2 * height)

    # img = io.imread('{}/{}_{:.4}.png'.format(save_dir, layer_to_visualize, name))
    # new_img = img[15:117, 25:180, :3]
    # new_img = exposure.equalize_adapthist(new_img, clip_limit=0.5)

    # plt.imshow(new_img)
    # plt.savefig('{}/HE_{}_{:.4}.png'.format(save_dir, layer_to_visualize, name))
    # io.imsave('{}/HE_{}_{:.4}.png'.format(save_dir, layer_to_visualize, name), new_img)

if __name__ == '__main__':
    if not os.path.isdir('./vis'):
        os.mkdir('./vis')
    # to_vis_layers=[6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 24]
    to_vis_layers = [9, 10, 11, 12, 13, 14, 15, 16]
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    save_dir = os.path.join('./vis', opt.name + '_' + opt.which_epoch)
    # print(save_dir)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    modulelist = model.netG.model._modules
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        input_A = model.input_A
        for vis_layer in to_vis_layers:
            visulize_all_channel_into_one(input_A, vis_layer, modulelist, save_dir)
            # filter_outputs(input_A, vis_layer, modulelist, save_dir)



