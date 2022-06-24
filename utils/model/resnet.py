import torch
import torch.nn as nn
from torchvision import models
import pdb


class ResNetModel(nn.Module):
    def __init__(self, out_dims=512, pretrained_weights='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\resnet_bert_models\\resnet18.pth') -> None:
        super(ResNetModel, self).__init__()
        self.out_dims = out_dims
        model = models.resnet18(pretrained=False)
        self.pretrained_weights = pretrained_weights

        # modify output
        self.backbone = torch.nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

    def load_pretrained_weights(self):
        net_dict = self.backbone.state_dict()
        pretrained_dict = torch.load(self.pretrained_weights)
        state_dict = {k: v for k, v in pretrained_dict.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        self.backbone.load_state_dict(net_dict)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)

        return x  # (batch_size, 512)

if __name__ == '__main__':
    model = ResNetModel(out_dims=512, pretrained_weights='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\resnet_bert_models\\resnet18.pth')
    inputs = torch.rand(9, 3, 224, 224)
    outputs = model(inputs)
