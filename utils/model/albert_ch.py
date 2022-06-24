from turtle import forward
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, BertConfig
from transformers import AlbertTokenizer, AlbertModel, AlbertConfig
from transformers import AutoTokenizer

import pdb


class AlBertModel(nn.Module):
    def __init__(self, out_dims=512, pretrained_name='voidful/albert_chinese_small') -> None:
        super(AlBertModel, self).__init__()
        self.out_dims = out_dims
        self.pretrained_name = pretrained_name
        self.backbone = None
        self.config = None
        self.out_layers = None

    def load_pretrained_weights(self):
        model = AlbertModel.from_pretrained(self.pretrained_name)
        config = AlbertConfig.from_pretrained(self.pretrained_name)
        self.backbone = model
        self.config = config
        self.out_layers = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.out_dims),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.pooler_output
        x = self.out_layers(x)

        return x  # (batch_size, 512)

if __name__ == '__main__':
    model = AlBertModel(out_dims=512,
                        pretrained_name='voidful/albert_chinese_small')
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

    input_ids=torch.tensor(tokenized_texts).view(-1, 20)
    output = model(input_ids)
