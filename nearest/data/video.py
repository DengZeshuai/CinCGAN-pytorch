import os
import glob
import numpy as np
from data import srdata
import scipy.misc as misc

class Video(srdata.SRData):
    def __init__(self, args, name='', train=True, benchmark=False):
        super(Video, self).__init__(
            args, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data)
        self.dir_hr = os.path.join(self.apath, 'DIV2K', 'DIV2K_train_HR')
        if self.train:
            self.dir_lr = os.path.join(self.apath, 'Video', self.args.video_train)
        else:
            self.dir_lr = os.path.join(self.apath, 'Video', self.args.video_test)
        self.dir_lrb = os.path.join(self.apath, 'DIV2K', 'DIV2K_train_LR_bicubic')
        self.ext = '.png'
    
    def _scan(self):
        list_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext))
        )

        list_lr =  [sorted(
                glob.glob(os.path.join(self.dir_lr, '*' + self.ext))
            ) for _ in self.scale]

        list_lrb = [[] for _ in self.scale]
        for  f in list_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):            
                list_lrb[si].append(os.path.join(
                    self.dir_lrb,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
        # print(len(list_hr), list_hr[0], len(list_lr), list_lr[0][0], len(list_lrb), list_lrb[0][0])
        return list_hr, list_lr, list_lrb
    
    def __len__(self):
        if self.train:
            return len(self.images_hr)
        return len(self.images_lr[0])

    def _get_index(self, idx):
        return idx
    
    def _load_file(self, idx):
        idx = self._get_index(idx)
        lr = self.images_lr[self.idx_scale][idx]
        lrb = self.images_lrb[self.idx_scale][idx]
        hr = self.images_hr[idx]
        if self.args.ext == 'img' or self.benchmark:
            filename = lr
            lr = misc.imread(lr)
            hr = misc.imread(hr)
            lrb = misc.imread(lrb)
        elif self.args.ext.find('sep') >= 0:
            filename = lr
            lr = np.load(lr)
            lrb = np.load(lrb)
            hr = np.load(hr)
        else:
            filename = str(idx + 1)

        filename = os.path.splitext(os.path.split(filename)[-1])[0]

        return lr, hr, lrb, filename