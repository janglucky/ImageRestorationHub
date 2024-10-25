import cv2
import os
import numpy as np

def show_cam_on_image(img, mask):
    heat_map = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    heat_map = np.float32(heat_map) / 255
    cam = heat_map + np.float32(img / 255)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

source_path = '/home/guider/data/sr/DDPD/test_c/source'
mask_path = '/home/guider/work/ImageRestorationHub/results/test_dafnet_ddpd_fourdata/visualization/DDPD_test_reverse' # 权重图
output_path = 'output'
source_files = sorted(os.listdir(source_path))
mask_files = sorted(os.listdir(mask_path))

for source, mask in zip(source_files, mask_files):

    source_file = os.path.join(source_path, source)
    mask_file = os.path.join(mask_path, mask)

    img_source = cv2.imread(source_file)
    img_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

    cam = show_cam_on_image(img_source, img_mask)

    cv2.imwrite(f'{output_path}/{os.path.basename(source)}', cam)

