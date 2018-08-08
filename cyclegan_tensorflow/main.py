import tensorflow as tf
import argparse
import os
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_dir',dest='dataset_dir',default='horse',help='the dataset of train and test')
parser.add_argument('--crop_size',dest='crop_size',default=256,help='the image size after cropped')
parser.add_argument('--input_channel',dest='input_channel',default=3,type=int,help='the channnel of inputy image')
parser.add_argument('--output_channel',dest='output_channel',default=3,type=int,help='the output of output image')
parser.add_argument('--batch_size',dest='batch_size',default=1,type=int,help='the image nummber in a batch')
parser.add_argument('--epoch',dest='epoch',default=20000,type=int,help='the training sum step')
parser.add_argument('--orginal_size',dest='orginal_size',default=296,type=int,help='the orginal image size')
parser.add_argument('--use_resnet',dest='use_resnet',type=bool,default=True,help='generation network use resnet reisidual block')
parser.add_argument('--use_lgan',dest='use_lgan',type=bool,default=True,help='gan loss defined in lsgan')
parser.add_argument('--ngf',dest='ngf',default=64,type=int,help='the num of gen filters in first conv layer')
parser.add_argument('--dgf',dest='dgf',default=64,type=int,help='the num of discriminator filters in first conv layer')
parser.add_argument('--phase',dest='phase',default='train',help='train test')
