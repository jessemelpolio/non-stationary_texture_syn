from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import random


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      'batch', not opt.no_dropout,
                                      self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def recurrent_test(self, step=5):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            rw = int(width/2)
            rh = int(height/2)
            self.real_A = Variable(self.fake_B.data[:, :, rh:rh + height, rw:rw + width], volatile=True)
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_A'.format(i), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

    def l2_searching(self, input_src, search_plate, stride=1):
        input_src_height, input_src_width = input_src.cpu().size()[2], input_src.cpu().size()[3]
        search_plate_height, search_plate_width = search_plate.cpu().size()[2], search_plate.cpu().size()[3]
        # print(type(input_src.data))
        searching_width = int((search_plate_width - input_src_width) / stride)
        searching_height = int((search_plate_height - input_src_height) / stride)
        min_loss = None
        min_w = 0
        min_h = 0
        for w_step in range(searching_width):
            for h_step in range(searching_height):
                # print(input_src_width, input_src_height)
                # print(search_plate_width, search_plate_height)
                # print(w_step * stride, h_step * stride, w_step * stride + input_src_width, h_step * stride + input_src_height)
                tmp = search_plate.narrow(2, int(h_step * stride), input_src_height).narrow(3, int(w_step * stride), input_src_width)
                loss = F.mse_loss(tmp, input_src)
                if w_step == 0 and h_step == 0:
                    min_loss = loss
                else:
                    if min_loss.data[0] > loss.data[0]:
                        min_loss = loss
                        min_w = w_step
                        min_h = h_step
        # print(min_loss)
        return min_w * stride, min_h * stride


    def recurrent_test_l2_searching(self, step=5):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('l2_search_real_{}_A'.format(0), real_A))
        results.append(('l2_search_fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            rw, rh = self.l2_searching(self.real_A.clone(), self.fake_B.clone())
            print("end selection: ", rw, rh)
            self.real_A = Variable(self.fake_B.data[:, :, rh:rh + height, rw:rw + width], volatile=True)
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('l2_search_real_{}_{}_{}_A'.format(i, rw, rh), real_A))
            results.append(('l2_search_fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

    def random_crop(self, crop_patch=6):
        input_size = self.input_A.cpu().shape
        width, height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        src_fake_B = self.fake_B.clone()
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_A', real_A))
        results.append(('fake_{}_B'.format('src'), fake_B))
        for i in range(0, crop_patch):
            rw = random.randint(0, width)
            rh = random.randint(0, height)
            self.real_A = Variable(src_fake_B.data[:, :, rh:rh + height, rw:rw + width], volatile=True)
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_{}_{}_A'.format(i, rw, rh), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

    def random_crop_256x256(self, crop_patch=6):
        input_size = self.input_A.cpu().shape
        width, height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        real_A_src = self.real_A.clone()
        # self.fake_B = self.netG.forward(self.real_A)
        # src_fake_B = self.fake_B.clone()
        real_A = util.tensor2im(self.real_A.data)
        # fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_A', real_A))
        # results.append(('fake_{}_B'.format('src'), fake_B))
        for i in range(0, crop_patch):
            rw = random.randint(0, width - 256)
            rh = random.randint(0, height - 256)
            self.real_A = Variable(real_A_src.data[:, :, rh:rh + 256, rw:rw + 256], volatile=True)
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('256_real_{}_{}_{}_A'.format(i, rw, rh), real_A))
            results.append(('512_fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)


    def stress_test_up(self, step=5, crop_size=64):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        rw = random.randint(0, width - crop_size)
        rh = random.randint(0, height - crop_size)
        self.real_A = Variable(self.real_A.data[:, :, rh:rh + crop_size, rw:rw + crop_size], volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            # rw = int(width/2)
            # rh = int(height/2)
            self.real_A = Variable(self.fake_B.data, volatile=True)
            print(self.real_A.size())
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_A'.format(i), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

    def stress_test_up_center(self, step=5, crop_size=64):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        rw = int((width - crop_size)/2)
        rh = int((height - crop_size)/2)
        self.real_A = Variable(self.real_A.data[:, :, rh:rh + crop_size, rw:rw + crop_size], volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            rw = int(crop_size/2)
            rh = int(crop_size/2)
            self.real_A = Variable(self.fake_B.data[:, :, rh:rh + crop_size, rw:rw + crop_size], volatile=True)
            print(self.real_A.size())
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_A'.format(i), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

    def stress_test_up_origin(self, step=3):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        # rw = random.randint(0, width - 64)
        # rh = random.randint(0, height - 64)
        # self.real_A = Variable(self.real_A.data, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            # rw = int(width/2)
            # rh = int(height/2)
            self.real_A = Variable(self.fake_B.data, volatile=True)
            print(self.real_A.size())
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_A'.format(i), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)
