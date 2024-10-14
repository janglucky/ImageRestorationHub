import torch
import os
import argparse
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from basicsr.archs.ecbsr_arch import ECBSR, PlainSR, model_ecbsr_rep


parser = argparse.ArgumentParser(description='ECBSR convertor')

## paramters for ecbsr
parser.add_argument('--scale', type=int, default=2, help = 'scale for sr network')
parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
parser.add_argument('--c_ecbsr', type=int, default=32, help = 'channels of ecb')
parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
parser.add_argument('--act_type', type=str, default='relu', help = 'prelu, relu, splus, rrelu')
parser.add_argument('--pretrain', type=str, default='/home/guider/work/super_resolution/experiments/ecbsr_m4c32_x2_paired_irgb_relu/models/net_g_350000.pth', help = 'path of pretrained model')
# parser.add_argument('--pretrain', type=str, default='/home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_div2k/models/net_g_570000.pth', help = 'path of pretrained model')
parser.add_argument('--output_folder', type=str, default='sr_result/256', help = 'folder of saved image')
parser.add_argument('--inp_c', type=int, default=1, help = 'channel size of input data')
parser.add_argument('--inp_h', type=int, default=256, help = 'height of input data')
parser.add_argument('--inp_w', type=int, default=192, help = 'width of input data')
parser.add_argument('--img', default='/home/guider/work/super_resolution/crop.bmp', help='low resolution image filename.')
parser.add_argument('--mark', default=None, help='low resolution image filename.')

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    device = torch.device('cuda')
    ## definitions of model, loss, and optimizer
    model_ecbsr = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.inp_c).to(device)
    model_plain = PlainSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.inp_c).to(device)
    
    # if args.pretrain is not None:
    #     print("load pretrained model: {}!".format(args.pretrain))
    #     ckpt = torch.load(args.pretrain, map_location=device)
    #     model_ecbsr.load_state_dict(torch.load(args.pretrain, map_location=device)['params_ema'], strict=True)
    # else:
    #     raise ValueError('the pretrain path is invalud!')
    
    model_ecbsr_rep(model_ecbsr, model_plain)

    
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_plain, (args.inp_c, args.inp_h, args.inp_w), print_per_layer_stat=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    if os.path.isdir(args.img):
        for file in os.listdir(args.img):
            if args.mark != None and args.mark not in file:
                continue
            filename = os.path.join(args.img, file)
            if args.inp_c == 1:
                img = Image.open(filename).convert("L")
            else:
                img = Image.open(filename).convert('RGB')
            # img = cv2.resize(img, (args.inp_w, args.inp_h), interpolation = cv2.INTER_LINEAR)
            # print(img.shape)
            # cv2.imwrite(os.path.join(args.output_folder, "gray_" + os.path.basename(file)), img)
           
            transform_test = transforms.Compose([
            transforms.ToTensor(),
            ])

            img = transform_test(img).unsqueeze(0).to(device)
            print(img.shape)
            import time
            start_time = time.time()
            pred = model_plain(img)
            print(f"total elapsed: {(time.time() - start_time) * 1000}ms")
            pred = pred.cpu().detach().numpy().squeeze(1).transpose((1, 2, 0))
            cv2.imwrite(os.path.join(args.output_folder, os.path.basename(file[:-4]+".bmp")), pred)
    elif args.img[-3:] == 'bin':
        width = 512
        height = 384
        frames = np.fromfile(args.img, dtype=np.uint8) + 128
        frame_size = width * height
        count = int(frames.shape[0] / frame_size)
        template = f'{os.path.basename(args.img)[:-4]}' + '_{:04d}.jpg'
        for i in range(count):
            img = frames[i*frame_size: (i+1)*frame_size].reshape(height, width)
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                ])
            img = transform_test(Image.fromarray(img)).unsqueeze(0)
            import time
            start_time = time.time()
            pred = model_plain(img)
            print(f"total elapsed: {(time.time() - start_time) * 1000}ms")
            pred = pred.detach().numpy().squeeze(0).transpose((1, 2, 0))

            # print(os.path.join(args.output_folder, template.format(i)))
            cv2.imwrite(os.path.join(args.output_folder, template.format(i)), pred)

    else:
        if args.color == 1:
            img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(args.img, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, (args.inp_w, args.inp_h), interpolation= cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(args.output_folder, "gray_"+os.path.basename(args.img)), img)

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        ])
        if args.color == 1:
            img = transform_test(Image.fromarray(img)).unsqueeze(0).to(device)
        else:
            img = transform_test(Image.fromarray(img)).unsqueeze(1).to(device)

        import time
        start_time = time.time()
        print(img.shape)
        pred = model_plain(img)
        print(f"total elapsed: {(time.time() - start_time) * 1000}ms")
        if args.color == 1:
            pred = pred.detach().cpu().numpy().squeeze(0).transpose((1, 2, 0))
        else:
            pred = pred.detach().cpu().numpy().squeeze(1).transpose((1, 2, 0))
        print(pred.shape)
        cv2.imwrite(os.path.join(args.output_folder, os.path.basename(args.img)), pred)