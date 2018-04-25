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

if __name__ == '__main__':
    if not os.path.isdir('./vis'):
        os.mkdir('./vis')
    to_vis_layers=[6, 9, 10, 11, 12, 13, 14, 15, 16, 18, 21, 24]
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
            filter_outputs(input_A, vis_layer, modulelist, save_dir)



