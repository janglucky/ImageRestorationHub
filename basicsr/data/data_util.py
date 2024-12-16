import cv2
import numpy as np
import torch
from os import path as osp
from torch.nn import functional as F

from basicsr.data.transforms import mod_crop
from basicsr.utils import img2tensor, scandir
from sklearn.metrics.pairwise import cosine_similarity


def ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.
    It is called by func:`calculate_ssim`.
    Args:
        img1 (ndarray): Images with range [0, 1] with order 'HWC'.
        img2 (ndarray): Images with range [0, 1] with order 'HWC'.
    Returns:
        float: ssim result.
    """

    C1 = (0.01)**2
    C2 = (0.03)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    # 公式二计算
    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calc_nrss_score(img, kernel_size, sigma):
    blurred = cv2.GaussianBlur(img, (kernel_size,kernel_size), sigmaX=sigma, sigmaY=sigma)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.float64)
    blurred = blurred.astype(np.float64)
    laplacian_img = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    laplacian_blur = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    return ssim(laplacian_img, laplacian_blur)


def read_img_seq(path, require_mod_crop=False, scale=1, return_imgname=False):
    """Read a sequence of images from a given folder path.

    Args:
        path (list[str] | str): List of image paths or image folder path.
        require_mod_crop (bool): Require mod crop for each image.
            Default: False.
        scale (int): Scale factor for mod_crop. Default: 1.
        return_imgname(bool): Whether return image names. Defalt False.

    Returns:
        Tensor: size (t, c, h, w), RGB, [0, 1].
        list[str]: Returned image name list.
    """
    if isinstance(path, list):
        img_paths = path
    else:
        img_paths = sorted(list(scandir(path, full_path=True)))
    imgs = [cv2.imread(v).astype(np.float32) / 255. for v in img_paths]

    if require_mod_crop:
        imgs = [mod_crop(img, scale) for img in imgs]
    imgs = img2tensor(imgs, bgr2rgb=True, float32=True)
    imgs = torch.stack(imgs, dim=0)

    if return_imgname:
        imgnames = [osp.splitext(osp.basename(path))[0] for path in img_paths]
        return imgs, imgnames
    else:
        return imgs


def generate_frame_indices(crt_idx, max_frame_num, num_frames, padding='reflection'):
    """Generate an index list for reading `num_frames` frames from a sequence
    of images.

    Args:
        crt_idx (int): Current center index.
        max_frame_num (int): Max number of the sequence of images (from 1).
        num_frames (int): Reading num_frames frames.
        padding (str): Padding mode, one of
            'replicate' | 'reflection' | 'reflection_circle' | 'circle'
            Examples: current_idx = 0, num_frames = 5
            The generated frame indices under different padding mode:
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            reflection_circle: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        list[int]: A list of indices.
    """
    assert num_frames % 2 == 1, 'num_frames should be an odd number.'
    assert padding in ('replicate', 'reflection', 'reflection_circle', 'circle'), f'Wrong padding mode: {padding}.'

    max_frame_num = max_frame_num - 1  # start from 0
    num_pad = num_frames // 2

    indices = []
    for i in range(crt_idx - num_pad, crt_idx + num_pad + 1):
        if i < 0:
            if padding == 'replicate':
                pad_idx = 0
            elif padding == 'reflection':
                pad_idx = -i
            elif padding == 'reflection_circle':
                pad_idx = crt_idx + num_pad - i
            else:
                pad_idx = num_frames + i
        elif i > max_frame_num:
            if padding == 'replicate':
                pad_idx = max_frame_num
            elif padding == 'reflection':
                pad_idx = max_frame_num * 2 - i
            elif padding == 'reflection_circle':
                pad_idx = (crt_idx - num_pad) - (i - max_frame_num)
            else:
                pad_idx = i - num_frames
        else:
            pad_idx = i
        indices.append(pad_idx)
    return indices


def paired_paths_from_lmdb(folders, keys):
    """Generate paired paths from lmdb files.

    Contents of lmdb. Taking the `lq.lmdb` for example, the file structure is:

    lq.lmdb
    ├── data.mdb
    ├── lock.mdb
    ├── meta_info.txt

    The data.mdb and lock.mdb are standard lmdb files and you can refer to
    https://lmdb.readthedocs.io/en/release/ for more details.

    The meta_info.txt is a specified txt file to record the meta information
    of our datasets. It will be automatically created when preparing
    datasets by our provided dataset tools.
    Each line in the txt file records
    1)image name (with extension),
    2)image shape,
    3)compression level, separated by a white space.
    Example: `baboon.png (120,125,3) 1`

    We use the image name without extension as the lmdb key.
    Note that we use the same key for the corresponding lq and gt images.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
            Note that this key is different from lmdb keys.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    if not (input_folder.endswith('.lmdb') and gt_folder.endswith('.lmdb')):
        raise ValueError(f'{input_key} folder and {gt_key} folder should both in lmdb '
                         f'formats. But received {input_key}: {input_folder}; '
                         f'{gt_key}: {gt_folder}')
    # ensure that the two meta_info files are the same
    with open(osp.join(input_folder, 'meta_info.txt')) as fin:
        input_lmdb_keys = [line.split('.')[0] for line in fin]
    with open(osp.join(gt_folder, 'meta_info.txt')) as fin:
        gt_lmdb_keys = [line.split('.')[0] for line in fin]
    if set(input_lmdb_keys) != set(gt_lmdb_keys):
        raise ValueError(f'Keys in {input_key}_folder and {gt_key}_folder are different.')
    else:
        paths = []
        for lmdb_key in sorted(input_lmdb_keys):
            paths.append(dict([(f'{input_key}_path', lmdb_key), (f'{gt_key}_path', lmdb_key)]))
        return paths


def paired_paths_from_meta_info_file(folders, keys, meta_info_file, filename_tmpl):
    """Generate paired paths from an meta information file.

    Each line in the meta information file contains the image names and
    image shape (usually for gt), separated by a white space.

    Example of an meta information file:
    ```
    0001_s001.png (480,480,3)
    0001_s002.png (480,480,3)
    ```

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        meta_info_file (str): Path to the meta information file.
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    with open(meta_info_file, 'r') as fin:
        gt_names = [line.split(' ')[0] for line in fin]

    paths = []
    for gt_name in gt_names:
        basename, ext = osp.splitext(osp.basename(gt_name))
        input_name = f'{filename_tmpl.format(basename)}{ext}'
        input_path = osp.join(input_folder, input_name)
        gt_path = osp.join(gt_folder, gt_name)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def tripled_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 3, ('The len of folders should be 3 with [input_folder, gt_folder, add_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 3, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder, add_folder = folders
    input_key, gt_key, add_key = keys

    input_paths = sorted(list(scandir(input_folder)))
    gt_paths = sorted(list(scandir(gt_folder)))
    add_paths = sorted(list(scandir(add_folder)))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []

    for input_path, gt_path, add_path in zip(input_paths, gt_paths, add_paths):
        input_path = osp.join(input_folder, input_path)
        gt_path = osp.join(gt_folder, gt_path)
        add_path = osp.join(add_folder, add_path)
    # for gt_path in gt_paths:
    #     basename, ext = osp.splitext(osp.basename(gt_path))
    #     input_name = f'{filename_tmpl.format(basename)}{ext}'
    #     input_path = osp.join(input_folder, input_name)
    #     assert input_name in input_paths, (f'{input_name} is not in ' f'{input_key}_paths.')
    #     gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{add_key}_path', add_path)]))
    return paths

def four_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 4, ('The len of folders should be 3 with [input_folder, gt_folder, add_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 4, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder, add_folder, dep_folder = folders
    input_key, gt_key, add_key, dep_key = keys

    input_paths = sorted(list(scandir(input_folder)))
    gt_paths = sorted(list(scandir(gt_folder)))
    add_paths = sorted(list(scandir(add_folder)))
    dep_paths = sorted(list(scandir(dep_folder)))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []

    for input_path, gt_path, add_path, dep_path in zip(input_paths, gt_paths, add_paths, dep_paths):
        input_path = osp.join(input_folder, input_path)
        gt_path = osp.join(gt_folder, gt_path)
        add_path = osp.join(add_folder, add_path)
        dep_path = osp.join(dep_folder, dep_path)
    # for gt_path in gt_paths:
    #     basename, ext = osp.splitext(osp.basename(gt_path))
    #     input_name = f'{filename_tmpl.format(basename)}{ext}'
    #     input_path = osp.join(input_folder, input_name)
    #     assert input_name in input_paths, (f'{input_name} is not in ' f'{input_key}_paths.')
    #     gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path), (f'{add_key}_path', add_path), (f'{dep_key}_path', dep_path)]))
    return paths

def paired_paths_from_folder(folders, keys, filename_tmpl):
    """Generate paired paths from folders.

    Args:
        folders (list[str]): A list of folder path. The order of list should
            be [input_folder, gt_folder].
        keys (list[str]): A list of keys identifying folders. The order should
            be in consistent with folders, e.g., ['lq', 'gt'].
        filename_tmpl (str): Template for each filename. Note that the
            template excludes the file extension. Usually the filename_tmpl is
            for files in the input folder.

    Returns:
        list[str]: Returned path list.
    """
    assert len(folders) == 2, ('The len of folders should be 2 with [input_folder, gt_folder]. '
                               f'But got {len(folders)}')
    assert len(keys) == 2, ('The len of keys should be 2 with [input_key, gt_key]. ' f'But got {len(keys)}')
    input_folder, gt_folder = folders
    input_key, gt_key = keys

    input_paths = sorted(list(scandir(input_folder)))
    gt_paths = sorted(list(scandir(gt_folder)))
    assert len(input_paths) == len(gt_paths), (f'{input_key} and {gt_key} datasets have different number of images: '
                                               f'{len(input_paths)}, {len(gt_paths)}.')
    paths = []

    for input_path, gt_path in zip(input_paths, gt_paths):
        input_path = osp.join(input_folder, input_path)
        gt_path = osp.join(gt_folder, gt_path)
    # for gt_path in gt_paths:
    #     basename, ext = osp.splitext(osp.basename(gt_path))
    #     input_name = f'{filename_tmpl.format(basename)}{ext}'
    #     input_path = osp.join(input_folder, input_name)
    #     assert input_name in input_paths, (f'{input_name} is not in ' f'{input_key}_paths.')
    #     gt_path = osp.join(gt_folder, gt_path)
        paths.append(dict([(f'{input_key}_path', input_path), (f'{gt_key}_path', gt_path)]))
    return paths


def paths_from_folder(folder):
    """Generate paths from folder.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """

    paths = list(scandir(folder))
    paths = [osp.join(folder, path) for path in paths]
    return paths


def paths_from_lmdb(folder):
    """Generate paths from lmdb.

    Args:
        folder (str): Folder path.

    Returns:
        list[str]: Returned path list.
    """
    if not folder.endswith('.lmdb'):
        raise ValueError(f'Folder {folder}folder should in lmdb format.')
    with open(osp.join(folder, 'meta_info.txt')) as fin:
        paths = [line.split('.')[0] for line in fin]
    return paths


def generate_gaussian_kernel(kernel_size=13, sigma=1.6):
    """Generate Gaussian kernel used in `duf_downsample`.

    Args:
        kernel_size (int): Kernel size. Default: 13.
        sigma (float): Sigma of the Gaussian kernel. Default: 1.6.

    Returns:
        np.array: The Gaussian kernel.
    """
    from scipy.ndimage import filters as filters
    kernel = np.zeros((kernel_size, kernel_size))
    # set element at the middle to one, a dirac delta
    kernel[kernel_size // 2, kernel_size // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter
    return filters.gaussian_filter(kernel, sigma)


def duf_downsample(x, kernel_size=13, scale=4):
    """Downsamping with Gaussian kernel used in the DUF official code.

    Args:
        x (Tensor): Frames to be downsampled, with shape (b, t, c, h, w).
        kernel_size (int): Kernel size. Default: 13.
        scale (int): Downsampling factor. Supported scale: (2, 3, 4).
            Default: 4.

    Returns:
        Tensor: DUF downsampled frames.
    """
    assert scale in (2, 3, 4), f'Only support scale (2, 3, 4), but got {scale}.'

    squeeze_flag = False
    if x.ndim == 4:
        squeeze_flag = True
        x = x.unsqueeze(0)
    b, t, c, h, w = x.size()
    x = x.view(-1, 1, h, w)
    pad_w, pad_h = kernel_size // 2 + scale * 2, kernel_size // 2 + scale * 2
    x = F.pad(x, (pad_w, pad_w, pad_h, pad_h), 'reflect')

    gaussian_filter = generate_gaussian_kernel(kernel_size, 0.4 * scale)
    gaussian_filter = torch.from_numpy(gaussian_filter).type_as(x).unsqueeze(0).unsqueeze(0)
    x = F.conv2d(x, gaussian_filter, stride=scale)
    x = x[:, :, 2:-2, 2:-2]
    x = x.view(b, t, c, x.size(2), x.size(3))
    if squeeze_flag:
        x = x.squeeze(0)
    return x
