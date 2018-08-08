from __future__ import division
import os
import tensorflow as tf
from module import *
from collections import namedtuple
class cylgan(object):
    def __init__(self,args):
        self.crop_size = args.crop_size
        self.input_channel = args.input_channel
        self.output_channel = args.output_channel
        self.batch_size = args.batch_size
        self.epoch = args.epoch
        self.orginal_size = args.orginal_size
        self.discriminator = discrminator
        if args.use_resnet:
            self.generator = generator_resnet
        else:
            self.generator = generator_unet
        if args.use_lgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion

        OPTIONS = namedtuple('OPTIONS',['batch_size','image_size','gf_dim','df_dim','output_c_dim','is_training'])
        self.options = OPTIONS._make([args.batch_size,args.crop_size,args.gnf,args.dnf,args.output_channel,args.phase=='train'])
        self._build_model()
        self.saver = tf.train.Saver()
        #self.pool =
    def _build_model(self):
        self.real_data = tf.placeholder(dtype=tf.float32,
                                        shape=[None,self.crop_size,self.crop_size,self.input_channel+self.output_channel],
                                        name='real_A_and_B_images')
        self.real_A = self.real_data[:,:,:,:self.input_channel]
        self.real_B = self.real_data[:,:,:,self.input_channel:self.input_channel+self.output_channel]
        self.fake_B = self.generator(self.real_A,self.options,False,name='geneatorAtoB')
        self.fake_A_ = self.generator(self.fake_B,self.options,False,name='generatorBtoA')
        self.fake_A = self.generator(self.real_B,self.options,True,name='generatorBtoA')


