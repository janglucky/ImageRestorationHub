# flake8: noqa
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import os.path as osp
from basicsr.train import train_pipeline

import ir.archs
import ir.data
import ir.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
