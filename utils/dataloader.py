import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

import os
import pdb


class TaoBao2022(Dataset):
    def __init__(self, img_dirs, alt_labels, gt_labels=None, datadir=None, dataset='train', ret_Image=False) -> None:
        super(TaoBao2022, self).__init__()
        self.img_dirs = img_dirs
        self.alt_labels = alt_labels
        if dataset=='train':
            self.gt_labels = gt_labels
        self.datadir = datadir
        self.mode = dataset
        self.ret_Image = ret_Image

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):
        assert 0 <= index < len(self.img_dirs)
        shop_id = list(self.img_dirs.keys())[index]
        img_folder_dir = os.path.join(self.datadir, shop_id)
        batch_size = len(self.img_dirs[shop_id])
        imgs = []  # list
        alt_texts = self.alt_labels[shop_id]  # list
        if self.mode == 'train':
            gt_texts = []
        for idx, img_file_name in enumerate(self.img_dirs[shop_id]):
            imgs.append(np.array(Image.open(os.path.join(img_folder_dir, img_file_name))))
            if self.mode == 'train':
                gt_texts.append(self.gt_labels[shop_id][idx][img_file_name])

        if self.ret_Image:
            return imgs, shop_id
        if self.mode == 'train':
            return imgs, alt_texts, gt_texts
        else:
            return imgs, alt_texts


class TaoBaoFeat2022(Dataset):
    def __init__(self, data_dirs) -> None:
        super(TaoBaoFeat2022, self).__init__()
        self.data_dirs = data_dirs

    def __len__(self):
        return len(self.data_dirs)

    def __getitem__(self, index):
        assert 0 <= index < len(self.img_dirs)
        shop_id = list(self.img_dirs.keys())[index]
        img_folder_dir = os.path.join(self.data_dirs, shop_id)
        imgs_feats = np.load(os.path.join(img_folder_dir, 'img_feat.npy'))

        return imgs_feats
