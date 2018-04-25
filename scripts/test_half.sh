#!/usr/bin/env bash
python test.py --dataroot ./datasets/fake --name 6_half_flip_30x30 --model test --which_epoch 100000 --which_model_netG resnet_2x_6blocks --which_model_netD n_layers --n_layers_D 4 --which_direction AtoB --dataset_mode single --norm batch --resize_or_crop none --fineSize 1100 --gpu_ids 7
