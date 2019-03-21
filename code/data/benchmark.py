import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Benchmark(srdata.SRData):
    def __init__(self, args, train=True):
        super(Benchmark, self).__init__(args, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]
        list_lrb = [[] for _ in self.scale]
        for entry in os.scandir(self.dir_hr):
            filename = os.path.splitext(entry.name)[0]
            list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
                list_lrb[si].append(os.path.join(
                    self.dir_lrb,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))

        list_hr.sort()
        for si, s in enumerate(self.scale):
            list_lr[si].sort()
            list_lrb[si].sort()

        return list_hr, list_lr, list_lrb

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR_{}'.format(self.args.lr_downsample))
        self.dir_lrb = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png'
