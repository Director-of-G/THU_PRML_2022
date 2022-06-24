import json
import os
import pdb
from PIL import Image

import torch
import torch.nn.functional as F
from dataloader import TaoBao2022, TaoBaoFeat2022
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import clip
from copy import deepcopy
import argparse

def load_labels(filedir='E:\\DataSets\\PRML2022\\medium\\medium', dataset='train', language='en', indices=None):
    print(language)
    if language == 'ch':
        file_name = ('train' if (dataset == 'train' or dataset == 'val') else 'test') + '_all.json'
    elif language == 'en':
        file_name = ('train' if (dataset == 'train' or dataset == 'val') else 'test') + '_all_0623.json'
    img_dir_dict, alt_label_dict = {}, {}
    if dataset == 'train' or dataset == 'val':
        gt_label_dict = {}
    with open(os.path.join(filedir, file_name), 'rb') as f:
        data = json.load(f)
        for idx, shop_id in enumerate(data.keys()):
            if (dataset == 'train' or dataset == 'val') and indices is not None:
                if idx not in indices:
                    continue
            shop_data = data[shop_id]
            img_dir_dict[shop_id] = [list(good_data.keys())[0] for good_data in shop_data['imgs_tags']]
            alt_label_dict[shop_id] = shop_data['optional_tags']
            if dataset == 'train' or dataset == 'val':
                gt_label_dict[shop_id] = shop_data['imgs_tags']

    if dataset == 'train' or dataset == 'val':
        return img_dir_dict, alt_label_dict, gt_label_dict
    else:
        return img_dir_dict, alt_label_dict

def load_data(filedir='E:\\DataSets\\PRML2022\\medium\\medium', args=None):
    # load train dataset
    img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir=filedir, dataset='train')
    train_dataset = TaoBao2022(img_dir_dict, alt_label_dict, gt_label_dict, os.path.join(filedir, 'train'), dataset='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    # load test dataset
    img_dir_dict, alt_label_dict= load_labels(filedir=filedir, dataset='test', language='en')
    test_dataset = TaoBao2022(img_dir_dict, alt_label_dict, datadir=os.path.join(filedir, 'test'), dataset='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, test_loader

def load_data_pure_feature(filedir='E:\\DataSets\\PRML2022\\medium\\medium', args=None):
    img_dir_dict, _, _ = load_labels(filedir=filedir, dataset='train')
    test_dataset = TaoBaoFeat2022(img_dir_dict)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return test_loader

def data_filtering(filedir='E:\\DataSets\\PRML2022\\medium\\medium', dataset='train'):
    input_file_name = dataset + '_all_en.json'
    output_file_name = dataset + '_all_en_rectified.json'
    data = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    with open(os.path.join(filedir, input_file_name), 'rb') as f:
        data = json.load(f)
        cnt = 0
        for shop_id in data.keys():
            cnt += 1
            print(cnt)
            shop_data = deepcopy(data[shop_id])
            optional_tags = deepcopy(shop_data['optional_tags'])
            optional_tags = [tag.lower() for tag in optional_tags]
            imgs_tags = [good_data[list(good_data.keys())[0]] for good_data in shop_data['imgs_tags']]
            imgs_tags = pd.DataFrame(imgs_tags).drop_duplicates()
            imgs_tags = np.array(imgs_tags).reshape(-1,).tolist()

            with torch.no_grad():
                # calculate text features
                optional_features = model.encode_text(clip.tokenize(optional_tags).to(device))
                imgtext_features = model.encode_text(clip.tokenize(imgs_tags).to(device))
                # normalization
                optional_features = F.normalize(optional_features, dim=1)
                imgtext_features = F.normalize(imgtext_features, dim=1)
                similarity_matrix = torch.matmul(imgtext_features, optional_features.T)
                assign_idx = torch.argmax(similarity_matrix, dim=1).cpu().numpy().tolist()     

            try:
                tag_dict = {imgs_tags[i]:optional_tags[assign_idx[i]] for i in range(len(imgs_tags))}
            except:
                pdb.set_trace()
            for good_idx in range(len(shop_data['imgs_tags'])):
                try:
                    key = list(shop_data['imgs_tags'][good_idx].keys())[0]
                    new_tag = tag_dict[shop_data['imgs_tags'][good_idx][key]]
                except:
                    pdb.set_trace()
                shop_data['imgs_tags'][good_idx][key] = new_tag
            shop_data['optional_tags'] = optional_tags
            data[shop_id] = shop_data
    pdb.set_trace()
    with open(os.path.join(filedir, output_file_name), 'wb') as f:
        json.dump(data, f)

def label_uniqueify(filedir='E:\\DataSets\\PRML2022\\medium\\medium'):
    file_name_train = 'train' + '_all.json'
    file_name_test = 'test' + '_all.json'
    tags_unified = []

    with open(os.path.join(filedir, file_name_train), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for shop_id in data.keys():
            shop_data = data[shop_id]
            opt_tags = shop_data['optional_tags']
            for tag in opt_tags:
                if tag not in tags_unified:
                    tags_unified.append(tag)

    with open(os.path.join(filedir, file_name_test), 'r', encoding='utf-8') as f:
        data = json.load(f)
        for shop_id in data.keys():
            shop_data = data[shop_id]
            opt_tags = shop_data['optional_tags']
            for tag in opt_tags:
                if tag not in tags_unified:
                    tags_unified.append(tag)

    pdb.set_trace()

def extract_feature(backbone=None,
                    preprocess=None,
                    filedir='E:\\DataSets\\PRML2022\\medium\\medium',
                    dataset='train'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dataset == 'train':
        img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir='E:\\DataSets\\PRML2022\\medium\\medium',
                                                                  dataset='train')
        for idx, pro_id in enumerate(list(img_dir_dict.keys())):
            print(idx)
            img_list = [Image.open(os.path.join(filedir, dataset, pro_id, ctg_id)) for ctg_id in img_dir_dict[pro_id]]
            img_list = [preprocess(image).unsqueeze(0).to(device) for image in img_list]
            batch_size = len(img_list)
            batch_feat = np.zeros((0, 768))
            for batch_idx in range(batch_size):
                with torch.no_grad():
                    feat = backbone.encode_image(img_list[batch_idx]).cpu()
                    feat /= feat.norm(dim=-1, keepdim=True)
                    batch_feat = np.concatenate((batch_feat, feat), axis=0)
            np.save(os.path.join(filedir, dataset, pro_id, 'img_feat.npy'), batch_feat)

# pre-processing of clothes labels
def label_pre_process():
    mapping_sheet_path = 'D:\\Tsinghua\\bachelor\\sem3_2\\prml\\project\\codes\\utils\\name_rectify.xls'
    mapping_sheet = pd.read_excel(mapping_sheet_path).to_numpy()
    mapping_sheet = {line[0]: line[1] for line in mapping_sheet}
    trans_sheet_path = 'D:\\Tsinghua\\bachelor\\sem3_2\\prml\\project\\codes\\utils\\name_trans.xls'
    trans_sheet = pd.read_excel(trans_sheet_path, nrows=4080).to_numpy()
    trans_sheet = {line[0]: line[1] for line in trans_sheet}

    read_json_path = 'E:\\DataSets\\PRML2022\\medium\\medium\\train_all.json'
    save_json_path = 'E:\\DataSets\\PRML2022\\medium\\medium\\train_all_0623.json'
    read_json = open(read_json_path, 'r', encoding='utf-8')
    read_json = json.load(read_json)
    for shop_idx, shop_id in enumerate(read_json.keys()):
        print('processing shop %d' % shop_idx)
        shop_data = read_json[shop_id]
        opt_tags = shop_data['optional_tags']
        opt_tags_new = []
        for tag in opt_tags:
            opt_tags_new.append(trans_sheet[mapping_sheet[tag]])
        shop_data['optional_tags'] = opt_tags_new
        imgs_tags = shop_data['imgs_tags']
        for idx, tag in enumerate(imgs_tags):
            key = list(tag.keys())[0]
            value = list(tag.values())[0]
            value_new = trans_sheet[mapping_sheet[value]]
            shop_data['imgs_tags'][idx][key] = value_new
        read_json[shop_id] = shop_data
        if len(np.unique(np.array(opt_tags_new)).tolist()) < len(opt_tags_new):
            pdb.set_trace()

    save_json = open(save_json_path, 'w')
    json.dump(read_json, save_json, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='number of sample images in a minibatch')
    # parser.add_argument('--num_workers', type=int, default=0,
    #                     help='num of threads for loading dataset')
    # train_loader, test_loader = load_data(filedir='E:\\DataSets\\PRML2022\\medium\\medium', args=parser.parse_args())
    # for idx, (imgs, alt_texts, gt_texts) in enumerate(train_loader):
    #     alt_texts = [text[0].lower() for text in alt_texts]
    #     gt_texts = [text[0].lower() for text in gt_texts]
    #     pdb.set_trace()

    # img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir='E:\\DataSets\\PRML2022\\medium\\medium', dataset='train')

    # data_filtering(filedir='E:\\DataSets\\PRML2022\\medium\\medium', dataset='train')

    # label_uniqueify(filedir='E:\\DataSets\\PRML2022\\medium\\medium')
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model, preprocess = clip.load("ViT-L/14@336px", device=device, download_root='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\clip_models')  # default:'~/.cache/clip'
    # extract_feature(backbone=model,
    #                 preprocess=preprocess,
    #                 filedir='E:\\DataSets\\PRML2022\\medium\\medium',
    #                 dataset='train')

    label_pre_process()
