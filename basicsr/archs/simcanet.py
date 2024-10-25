import torch
import torch.nn as nn


class SimpleCaNet(nn.Module):

    def __init__(self, in_channel, out_channel, feat_num):

        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)