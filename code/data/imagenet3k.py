import os
import glob
from data import srdata

class ImageNet3K(srdata.SRData):
    def __init__(self, args, name='ImageNet3K', train=True, benchmark=False):
        super(ImageNet3K, self).__init__(
            args, train=train, benchmark=benchmark
        )

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'ImageNet3K')
        self.dir_hr = os.path.join(self.apath, 'ImageNet3K_HR')
        self.dir_lr = os.path.join(self.apath, 'ImageNet3K_LR_{}'.format(self.args.lr_downsample))
        self.dir_lrb = os.path.join(self.apath, 'ImageNet3K_LR_bicubic')
        self.ext = '.png'
    
    def _scan(self):
        list_hr = sorted(
            glob.glob(os.path.join(self.dir_hr, '*' + self.ext))
        )

        list_lr = [[] for _ in self.scale]
        list_lrb = [[] for _ in self.scale]
    
        for  f in list_hr:
            filename, _ = os.path.splitext(os.path.basename(f))
            for si, s in enumerate(self.scale):
                list_lr[si].append(os.path.join(
                    self.dir_lr,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
                list_lrb[si].append(os.path.join(
                    self.dir_lrb,
                    'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                ))
        
        return list_hr, list_lr, list_lrb
    
    def __len__(self):
        return len(self.images_hr)

    def _get_index(self, idx):
        return idx