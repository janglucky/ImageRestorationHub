from basicsr.metrics.psnr_ssim import calculate_psnr, calculate_ssim
from basicsr.utils import img2tensor, imfrombytes
from basicsr.utils.file_client import HardDiskBackend
import os
import cv2
from tqdm import tqdm


pred_path = '/home/guider/work/ImageRestorationHub/inference/ddpd_train_coarse_full'
# pred_path = '/home/guider/nas/data/sr/DDPD/train_c/coarse'
targ_path = '/home/guider/nas/data/sr/DDPD/train_c/target'

pred_files = sorted(os.listdir(pred_path))
targ_files = sorted(os.listdir(targ_path))

size = len(pred_files)
psnr_total = 0.0
ssim_total = 0.0

file_client = HardDiskBackend()


for pred_file, targ_file in zip(pred_files, targ_files):

    pred_file = os.path.join(pred_path, pred_file)
    imgbytes = file_client(pred_file)
    pred = imfrombytes(imgbytes)

    targ_file = os.path.join(targ_path, targ_file)
    imgbytes = file_client(targ_file)
    targ = imfrombytes(imgbytes)



    psnr = calculate_psnr(pred, targ, 0, test_y_channel=False)
    ssim = calculate_ssim(pred, targ, 0, test_y_channel=False)
    print(f'#{os.path.basename(pred_file)} psnr: {psnr}, ssim: {ssim}')
    psnr_total += psnr
    ssim_total += ssim


print(f'Total psnr: {psnr_total / size} ssim: {ssim_total / size}')


# corase Total psnr: 25.115175047292137 ssim: 0.7427271640784422
# train sub Total psnr: 22.54837079162796 ssim: 0.6795700877291238
# train full Total psnr: 24.76196530412381 ssim: 0.746663451834118