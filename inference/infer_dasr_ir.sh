 CUDA_VISIBLE_DEVICES=1 python infer_dasr.py --scale 4 \
--pretrain_p /home/guider/work/DASR/experiments/train_DASR_x4/models/net_p_345000.pth \
--pretrain_g /home/guider/work/DASR/experiments/train_DASR_x4/models/net_g_345000.pth \
--inp_c 1 \
--use_bias \
--img /home/guider/work/DASR/inference/test_pic/benchmark_256x192_zp56a