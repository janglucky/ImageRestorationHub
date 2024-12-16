import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file, calc_nrss_score,tripled_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop, tripled_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CMoeImageDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(CMoeImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder, self.cond_folder = opt['dataroot_gt'], opt['dataroot_lq'], opt['dataroot_cond']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            # self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl)
            self.paths = tripled_paths_from_folder([self.lq_folder, self.gt_folder, self.cond_folder], ['lq', 'gt', 'cond'], self.filename_tmpl)

    def calc_cls_label(self, img, size = 64, level = 3):
        """
        patch_size 最小分块大小
        level 块尺度级别
        """
        h, w = img.shape[:2]
        labels = [ np.zeros((h // (size * 2 **i), w //(size * 2 ** i)), dtype=np.int64) for i in range(level)]

        for lvl in range(level):
            ps = size * 2 ** lvl
            for i in range(h // ps):
                for j in range(w // ps):
                    patch = img[i*ps: (i+1)*ps, j*ps:(j+1)*ps]
                    score = calc_nrss_score(patch, 11, 3.6)
                    if score < 0.25:
                        labels[lvl][i][j] = 0
                    elif score < 0.5:
                        labels[lvl][i][j] = 1
                    elif score < 0.8:
                        labels[lvl][i][j] = 2
                    else:
                        labels[lvl][i][j] = 3
        return labels

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        cond_path = self.paths[index]['cond_path']
        img_bytes = self.file_client.get(cond_path, 'cond')
        img_cond = imfrombytes(img_bytes, float32=True)

        

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            if gt_size > 0:
                # random crop
                img_gt, img_lq, img_cond = tripled_random_crop(img_gt, img_lq, img_cond, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq, img_cond = augment([img_gt, img_lq, img_cond], self.opt['use_flip'], self.opt['use_rot'])

        labels = self.calc_cls_label(img_cond, 64, 4)
        # TODO: color space transform
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq, img_cond = img2tensor([img_gt, img_lq, img_cond], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            normalize(img_cond, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'cond': img_cond, 'lq_path': lq_path, 'gt_path': gt_path, 'cond_path': cond_path, 'labels': labels}

    def __len__(self):
        return len(self.paths)
