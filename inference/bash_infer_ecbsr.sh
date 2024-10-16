python infer_ecbsr.py \
--scale 1 \
--inp_c 3 \
--m_ecbsr 8 \
--c_ecbsr 32 \
--idt_ecbsr 0 \
--act_type relu \
--output_folder ddpd_train_coarse_full \
--img /home/guider/data/sr/DDPD/train_c/coarse \
--gt /home/guider/data/sr/DDPD/train_c/target \
--pretrain /home/guider/work/ImageRestorationHub/experiments/train_ecbsr_m8c32_x1_paired_ddpd_relu/models/net_g_7000.pth # 当前上机最新版
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m4c32_x2_paired_irgb_relu/models/net_g_4320000.pth # zg31a 实时超分

# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m4c32_x2_paired_irgb_relu/models/net_g_2295000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m4c16_x2_paired_irgb_relu/models/net_g_215000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_irgb_relu/models/net_g_180000.pth
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_div2k/models/net_g_570000.pth # 9.2日版本
# --pretrain /home/guider/work/super_resolution/experiments/ecbsr_m8c32_x2_paired_irgb_relu/models/net_g_20000.pth