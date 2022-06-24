import sys
from turtle import forward
sys.path.append('./utils/model')

import torch
import torch.nn as nn
from resnet import ResNetModel
from albert_ch import AlBertModel
from transformers import BertTokenizer
import clip

import pdb

"""
    HandCraft Model With ResNet18 and AlBert
"""

class MatchNet(nn.Module):
    def __init__(self, feat_dims) -> None:
        super(MatchNet, self).__init__()
        self.feat_dims = feat_dims
        self.relu = nn.ReLU()

    def forward(self, imgs_feats, texts_feats):
        # img_feats => (img_batch_size, feat_dims)
        # texts_feats => (text_batch_size, feat_dims)

        # L2 normalization
        imgs_feats = self.relu(imgs_feats)
        texts_feats = self.relu(texts_feats)
        imgs_feats = imgs_feats / torch.norm(imgs_feats, p=2)
        texts_feats = texts_feats / torch.norm(texts_feats, p=2)

        # cosine distance similarity measurement
        similarity = torch.matmul(imgs_feats, texts_feats.T)  # (img_batch_size, text_batch_size)
        # return similarity as probits
        return similarity

class MultiModalClassifier(nn.Module):
    def __init__(self, feat_dims) -> None:
        super(MultiModalClassifier, self).__init__()
        self.feat_dims = feat_dims
        self.text_backbone = AlBertModel(out_dims=feat_dims, pretrained_name='voidful/albert_chinese_small')
        self.img_backbone = ResNetModel(out_dims=feat_dims, pretrained_weights='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\resnet_bert_models\\resnet18.pth')
        self.match_net = MatchNet(feat_dims=feat_dims)

    def load_pretrained_weights(self):
        self.img_backbone.load_pretrained_weights()
        self.text_backbone.load_pretrained_weights()

    def forward(self, images, texts):
        # feature extractor
        img_feats = self.img_backbone(images)  # (img_batch_size, feat_dims)
        text_feats = self.text_backbone(texts)  # (text_batch_size, feat_dims)

        # calculate similarity as probits
        similarity = self.match_net(img_feats, text_feats)

        return similarity

"""
    Pretrained Model With CLIP
"""
def load_clip_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32",
                                  device=device,
                                  jit=False,
                                  download_root='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\clip_models')
    if device == 'cpu':
        model.float()
    else:
        clip.model.convert_weights(model)

    return model, preprocess
    

class CLIPClassifier(nn.Module):
    def __init__(self, clip_model=None) -> None:
        super(CLIPClassifier, self).__init__()
        self.backbone = clip_model

    def forward(self, images, texts):
        # get logits
        logits_per_image, logits_per_text = self.backbone(images, texts)

        return logits_per_image, logits_per_text


if __name__ == '__main__':
    model = MultiModalClassifier(feat_dims=512)
    model.load_pretrained_weights()
    
    tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_small')
    input_texts = ['藏青色', '蓝色', '西瓜红']
    tokenized_texts = []
    for text in input_texts:
        tokenized_text=tokenizer.encode(text,
                                        max_length=20,
                                        padding='max_length',
                                        truncation=True)
        tokenized_texts.append(tokenized_text)

    input_texts = torch.tensor(tokenized_texts).view(-1, 20)
    input_images = torch.rand(9, 3, 224, 224)
    output = model(input_images, input_texts)
