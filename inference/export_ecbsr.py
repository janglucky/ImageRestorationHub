import os
import argparse
import torch
from basicsr.archs.ecbsr_arch import ECBSR, PlainSR, model_ecbsr_rep
from basicsr.utils import misc

def parse_args():
    parser = argparse.ArgumentParser(description='ECBSR convertor')

    ## paramters for ecbsr
    # about model architecture
    parser.add_argument('--scale', type=int, default=2, help = 'scale for sr network')
    parser.add_argument('--m_ecbsr', type=int, default=4, help = 'number of ecb')
    parser.add_argument('--c_ecbsr', type=int, default=4, help = 'channels of ecb')
    parser.add_argument('--idt_ecbsr', type=int, default=0, help = 'incorporate identity mapping in ecb or not')
    parser.add_argument('--act_type', type=str, default='prelu', help = 'prelu, relu, splus, rrelu')
    parser.add_argument('--inp_c', type=int, default=1, help = 'channel size of input data')
    parser.add_argument('--inp_h', type=int, default=512, help = 'height of input data')
    parser.add_argument('--inp_w', type=int, default=640, help = 'width of input data')

    # export param
    parser.add_argument('--name', type=str, default='v0.0.0', help = 'prelu, relu, splus, rrelu')
    parser.add_argument('--version', type=str, default='v0.0.0', help = 'prelu, relu, splus, rrelu')
    parser.add_argument('--pretrain', type=str, default=None, help = 'path of pretrained model')
    parser.add_argument('--target_frontend', type=str, default='onnx', help = 'onnx, torch')
    parser.add_argument('--output_folder', type=str, default='./', help = 'output folder')
    parser.add_argument('--is_dynamic_shape', type=int, default=0, help = 'dynamic batches or not')
    parser.add_argument('--export_norm', type=int, default=0, help = 'export output to [0-1]')
    parser.add_argument('--opset', type=int, default=18, help = 'width of input data')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    
    device = torch.device('cpu')
    ## definitions of model, loss, and optimizer
    model_ecbsr = ECBSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, with_idt=args.idt_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.inp_c).to(device)
    model_plain = PlainSR(module_nums=args.m_ecbsr, channel_nums=args.c_ecbsr, act_type=args.act_type, scale=args.scale, colors=args.inp_c, export_norm=args.export_norm).to(device)
    
    if args.pretrain is not None:
        print("load pretrained model: {}!".format(args.pretrain))
        model_ecbsr.load_state_dict(torch.load(args.pretrain, map_location=device)['params_ema'], strict=True)
    else:
        raise ValueError('the pretrain path is invalud!')

    model_ecbsr_rep(model_ecbsr, model_plain)

    ## convert model to onnx
    output_name = f'{args.version}.{misc.get_time_str()}'
    if args.target_frontend == 'onnx':
        output_name = os.path.join(args.output_folder, f'{args.name}_op{args.opset}_ecbsr_x{args.scale}_m{args.m_ecbsr}c{args.c_ecbsr}_{args.act_type}_{args.inp_w}x{args.inp_h}_{output_name}' + '.onnx')
        fake_x = torch.rand(1, args.inp_c, args.inp_h, args.inp_w, requires_grad=False)

        dynamic_params = None
        if args.is_dynamic_shape:
            dynamic_params = {'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}

        torch.onnx.export(
            model_plain, 
            fake_x, 
            output_name, 
            export_params=True, 
            opset_version=args.opset, 
            do_constant_folding=True, 
            input_names=['input'], 
            output_names=['output'],
            dynamic_axes=dynamic_params
        )
        
    elif args.target_frontend == 'torch':
        output_name = os.path.join(args.output_folder, f'{args.name}_ecbsr_x{args.scale}_m{args.m_ecbsr}c{args.c_ecbsr}_{args.act_type}_{args.inp_w}x{args.inp_h}_{output_name}' + '.pt')
        fake_x = torch.rand(1, args.inp_c, args.inp_h, args.inp_w, requires_grad=False)
        trace_model = torch.jit.trace(model_plain, fake_x)
        trace_model.save(output_name)
    else:
        raise ValueError('invalid type of frontend!')