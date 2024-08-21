# flake8: noqa
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp
from basicsr.train import train_pipeline

import isr.archs
import isr.data
import isr.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
