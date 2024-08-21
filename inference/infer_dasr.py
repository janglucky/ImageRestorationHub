import torch
import os
import argparse
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from basicsr.archs.srresnetdynamic_arch import MSRResNetDynamic
from basicsr.archs.degradation_prediction_arch import Degradation_Predictor


parser = argparse.ArgumentParser(description='ECBSR convertor')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=4, help = 'scale for sr network')
parser.add_argument('--num_feat', type=int, default=64, help = 'feature dims')
parser.add_argument('--num_block', type=int, default=16, help = 'block num')
parser.add_argument('--num_models', type=int, default=5, help = 'num of experts')
parser.add_argument('--nf', type=int, default=64, help = 'num of experts')
parser.add_argument('--num_params', type=int, default=33, help = 'num of degradation params')
parser.add_argument('--pretrain_g', type=str, default='net_g.pth', help = 'path of pretrained g model')
parser.add_argument('--pretrain_p', type=str, default='net_p.pth', help = 'path of pretrained p model')
parser.add_argument('--output_folder', type=str, default='output', help = 'folder of saved image')
parser.add_argument('--inp_c', type=int, default=1, help = 'channel size of input data')
parser.add_argument('--use_bias', action='store_true', help = 'use bias')
parser.add_argument('--img', default='img_dir', help='low resolution image filename.')

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    device = torch.device('cuda')
    ## definitions of model, loss, and optimizer
    model_g = MSRResNetDynamic(args.inp_c, args.inp_c, args.num_feat, args.num_block, args.num_models, args.scale)
    model_p = Degradation_Predictor(args.inp_c, args.nf, args.num_params, args.num_models, args.use_bias)
    
    if args.pretrain_p is not None and args.pretrain_g is not None:
        print("load pretrained model: {}, {}!".format(args.pretrain_g, args.pretrain_p))
        model_g.load_state_dict(torch.load(args.pretrain_g, map_location=device)['params'], strict=True)
        model_p.load_state_dict(torch.load(args.pretrain_p, map_location=device)['params'], strict=True)
    else:
        raise ValueError('the pretrain path is invalud!')

    model_g.eval()
    model_p.eval()
    model_g.to(device)
    model_p.to(device)

    if os.path.isdir(args.img):
        for file in os.listdir(args.img):
            filename = os.path.join(args.img, file)
            if args.inp_c == 1:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(filename, cv2.IMREAD_COLOR)

            # img = img[:512,:640]
            # cv2.imwrite(os.path.join(args.output_folder, "gray_"+os.path.basename(filename)), img)
            transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])
            img = transform_test(Image.fromarray(img)).unsqueeze(0).to(device)

            import time
            start_time = time.time()
            predicted_params, weights = model_p(img)
            print(predicted_params)
            print(weights.shape)
            pred = model_g(img, weights)
            pred = pred * 255
            pred = pred.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
            print(f"total elapsed: {(time.time() - start_time) * 1000}ms")
            cv2.imwrite(os.path.join(args.output_folder, os.path.basename(filename)), pred)
            
    else:
        img = cv2.imread(args.img, cv2.IMREAD_COLOR)

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
        img = transform_test(Image.fromarray(img)).unsqueeze(0)

        import time
        start_time = time.time()
        predicted_params, weights = model_p(img)
        print(predicted_params)
        print(weights.shape)
        pred = model_g(img, weights)
        pred = pred * 255
        print(f"total elapsed: {(time.time() - start_time) * 1000}ms")
        pred = pred.cpu().detach().numpy().squeeze(0).transpose((1, 2, 0))
        cv2.imwrite(os.path.join(args.output_folder, os.path.basename(args.img)), pred)