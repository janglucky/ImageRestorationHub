python export_ecbsr.py \
--idt_ecbsr 0 \
--act_type relu \
--target_frontend onnx \
--output_folder output \
--is_dynamic_shape 0 \
--m_ecbsr 8 \
--c_ecbsr 32 \
--scale 2 \
--inp_c 1 \
--inp_h 288 \
--inp_w 384 \
--name zp56a \
--name version \
--opset 12 \
--export_norm 0 \
--pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_div2k_relu/models/net_g_900000.pth # 最新 拍照超分
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m4c32_x2_paired_irgb_relu/models/net_g_4320000.pth # zg31a 实时超分
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_div2k/models/net_g_570000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x4_cnn_deblur/models/net_g_200000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_cnn/models/net_g_2260000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m4c16_x2_cnn/models/net_g_240000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_cnn/models/net_g_690000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_cnn/models/net_g_190000.pth