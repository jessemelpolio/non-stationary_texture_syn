import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import torch.nn as nn
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .vgg import VGG, GramMatrix, GramMSELoss


class HalfGanStyleModel(BaseModel):
    def name(self):
        return 'HalfGanStyleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   int(opt.fineSize / 2), int(opt.fineSize / 2))
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)


        self.style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
        # self.content_layers = ['r42']
        self.loss_layers = self.style_layers
        self.loss_fns = [GramMSELoss()] * len(self.style_layers)
        if torch.cuda.is_available():
            self.loss_fns = [loss_fn.cuda() for loss_fn in self.loss_fns]
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(os.getcwd() + '/Models/' + 'vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

        print(self.vgg.state_dict().keys())

        self.style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
        # self.content_weights = [1e0]
        self.weights = self.style_weights

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.start_points= input['A_start_point']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # TODO here we use real image to create fake_AB
        fake_AB = self.fake_AB_pool.query(self.fake_B.clone())
        # fake_AB = self.fake_AB_pool.query(torch.cat((self.real_B, self.fake_B), 1))
        self.pred_fake = self.netD.forward(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_AB = self.real_B.clone()
        self.pred_real = self.netD.forward(real_AB)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        if self.opt.use_style:
            style_targets = [GramMatrix()(A).detach() for A in self.vgg(self.real_B, self.style_layers)]
            # content_targets = [A.detach() for A in self.vgg(self.real_B, self.content_layers)]
            targets = style_targets
            out = self.vgg(self.fake_B, self.loss_layers)
            layer_losses = [self.weights[a] * self.loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
            # print(layer_losses)
            loss = sum(layer_losses)
            self.style_loss = loss
            loss.backward(retain_graph=True)
            self.style_loss_value = self.style_loss.data[0]
        else:
            self.style_loss_value = 0

        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B.clone()
        pred_fake = self.netD.forward(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        # self.loss_G = self.loss_G_GAN

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        # print(self.pred_real)
        # print(self.pred_fake)
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ('Style', self.style_loss_value)
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)]), self.start_points

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
