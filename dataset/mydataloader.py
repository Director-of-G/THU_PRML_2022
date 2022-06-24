import clip
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from transformers import BertTokenizer, BertModel, BertConfig
from PIL import Image
import numpy as np

import os
import pdb
import sys
file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_path, '..\\utils'))
from preprocessing import load_labels


def get_masked_image(image: Image=None, dir: str=None, ret_mask=True):
    # the mask is in the same size of the origin image
    index = int(dir.split('.')[0][-1])  # the order of current image in the product folder
    dirname = os.path.dirname(dir)
    maskpath = os.path.join(dirname, 'mask_%d.npz' % index)

    mask = np.load(maskpath)
    mask = mask['mask']

    if ret_mask:
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        mask = transform(Image.fromarray(mask)).unsqueeze(0)
        return mask

    image = np.array(image) * mask
    image = Image.fromarray(image)

    return image

class TaoBao2022_Img(Dataset):
    def __init__(self, args=None, img_dirs=None, datadir=None, preprocess=None) -> None:
        super(TaoBao2022_Img, self).__init__()
        self.args = args
        self.img_dirs = img_dirs
        self.datadir = datadir

        if args.for_clip:  # use clip preprocessing strategies
            self.transform = preprocess

        else:  # use preprocessing strategies for ImageNet
            self.transform = transforms.Compose([
                transforms.CenterCrop(224), # (224, 224)
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.img_dirs)

    def __getitem__(self, index):
        assert 0 <= index < len(self.img_dirs)

        shop_id = list(self.img_dirs.keys())[index]
        img_folder_dir = os.path.join(self.datadir, shop_id)
        imgs = torch.zeros(0, 3, 224, 224)

        for idx, img_file_name in enumerate(self.img_dirs[shop_id]):
            img_dir = os.path.join(img_folder_dir, img_file_name)
            img = Image.open(img_dir)
            img = self.transform(img).view(1, 3, 224, 224)  # Tensor: (1, 224, 224)
            if self.args.use_mask:
                mask = get_masked_image(dir=img_dir, 
                                        ret_mask=True)  # Tensor: (1, 1, 224, 224)
                img = img * mask  # Tensor: (1, 3, 224, 224)
            imgs = torch.cat((imgs, img), dim=0)

        return imgs.float()  # Tensor: (batch_size, 224, 224)

class TaoBao2022_Text(Dataset):
    def __init__(self, args=None, img_dict=None, alt_labels=None, gt_labels=None, mode='train', max_length=20) -> None:
        super(TaoBao2022_Text, self).__init__()
        self.args = args
        self.img_dict = img_dict
        self.alt_labels = alt_labels
        self.mode = mode
        self.max_length = max_length
        if mode=='train':
            self.gt_labels = gt_labels

        # tokenizer
        if args.for_clip:
            self.tokenizer = clip.tokenize
        else:
            pretrained = 'voidful/albert_chinese_small'  # tiny albert model
            self.tokenizer = BertTokenizer.from_pretrained(pretrained)

    def __len__(self):
        return len(self.alt_labels)

    def __getitem__(self, index):
        assert 0 <= index < len(self.alt_labels)

        shop_id = list(self.alt_labels.keys())[index]
        batch_size = len(self.img_dict[shop_id])

        # tokenzie alternative labels
        alt_texts = self.alt_labels[shop_id]  # list of alternative texts
        alt_tokens = torch.zeros(0, self.max_length)
        if self.args.for_clip:
            alt_tokens = self.tokenizer(alt_texts).view(-1, 77)
        else:
            for text in alt_texts:
                tokenized_text = self.tokenizer.encode(text,
                                                    max_length=self.max_length,
                                                    padding='max_length',
                                                    truncation=True)  # list
                tokenized_text = torch.tensor(tokenized_text).view(1, self.max_length)  # (1, 30)
                alt_tokens = torch.cat((alt_tokens, tokenized_text), dim=0)
        
        if self.mode == 'train':
            gt_labels = []
            gt_texts = self.gt_labels[shop_id]
            for text in [list(text.values())[0] for text in gt_texts]:
                gt_labels.append(alt_texts.index(text))
            gt_labels = torch.tensor(gt_labels).view(-1, 1)
            
            return alt_tokens.float(), [[shop_id]], [[batch_size]], gt_labels.int()
        
        else:
            return alt_tokens.float(), [[shop_id]], [[batch_size]]

def my_collate_fn_img(batch):
    return torch.vstack(batch)

def my_collate_fn_text(batch):
    alt_tokens = [item[0] for item in batch]
    shop_ids = [item[1] for item in batch]
    batch_sizes = [item[2] for item in batch]
    if len(batch[0]) == 4:
        gt_labels = [item[3] for item in batch]
        return [torch.vstack(alt_tokens), np.concatenate(shop_ids).tolist(), np.concatenate(batch_sizes).tolist(), torch.vstack(gt_labels)]
    else:
        return [torch.vstack(alt_tokens), np.concatenate(shop_ids).tolist(), np.concatenate(batch_sizes).tolist()]

if __name__ == '__main__':
    filedir = 'E:\\DataSets\\PRML2022\\medium\\medium'
    traintest = 'train'
    img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir=filedir,
                                                              dataset=traintest)
    
    # dataset_img = TaoBao2022_Img(img_dirs=img_dir_dict,
    #                              datadir=os.path.join(filedir, 'train'))
    # dataloader_img = DataLoader(dataset=dataset_img,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=0,
    #                             collate_fn=my_collate_fn_img)

    dataset_text = TaoBao2022_Text(img_dict=img_dir_dict,
                                   alt_labels=alt_label_dict,
                                   gt_labels=gt_label_dict,
                                   mode=traintest,
                                   max_length=20)
    dataloader_text = DataLoader(dataset=dataset_text,
                                 batch_size=2,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=my_collate_fn_text)

    for idx, data in enumerate(dataloader_text):
        pdb.set_trace()
