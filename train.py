import argparse
from operator import mod
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import clip
from dataset.mydataloader import TaoBao2022_Img, TaoBao2022_Text, my_collate_fn_img, my_collate_fn_text
from utils.preprocessing import load_labels
from random import shuffle

from utils.model.classifier import MultiModalClassifier
from utils.criterion import TaoBaoMetric

import parser
from copy import deepcopy

import time
import os
import pdb

# global variables for tensorboard
global train_iters, eval_iters

def train_val_split(data_size, train_ratio=0.8):
    indices = np.arange(data_size)
    shuffle(indices)
    train_size = round(data_size * train_ratio)
    indices_train = indices[:train_size]
    indices_val = indices[train_size:]

    return indices_train, indices_val

def get_dataloaders(args=None, preprocess=None):
    # random seed
    seed = int(time.time()) % 12345
    # train val split
    data_size = args.train_size  # 21597
    indices_train, indices_val = train_val_split(data_size, train_ratio=0.8)
    # get training dataset
    img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir=args.filedir,
                                                              dataset='train',
                                                              language='en',
                                                              indices=indices_train)
    dataset_img_train = TaoBao2022_Img(args=args,
                                       img_dirs=img_dir_dict,
                                       datadir=os.path.join(args.filedir, 'train'),
                                       preprocess=preprocess)
    torch.manual_seed(seed=seed)
    gen = torch.Generator(device='cpu')
    dataloader_img_train = DataLoader(dataset=dataset_img_train,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=my_collate_fn_img,
                                      generator=gen)
    dataset_text_train = TaoBao2022_Text(args=args,
                                         img_dict=img_dir_dict,
                                         alt_labels=alt_label_dict,
                                         gt_labels=gt_label_dict,
                                         mode='train',
                                         max_length=20)
    torch.manual_seed(seed=seed)
    gen = torch.Generator(device='cpu')
    dataloader_text_train = DataLoader(dataset=dataset_text_train,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       collate_fn=my_collate_fn_text,
                                       generator=gen)
    train_loader = (dataloader_img_train, dataloader_text_train)

    # get validation dataset
    img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir=args.filedir,
                                                              dataset='val',
                                                              language='en',
                                                              indices=indices_val)
    dataset_img_val = TaoBao2022_Img(args=args,
                                     img_dirs=img_dir_dict,
                                     datadir=os.path.join(args.filedir, 'train'),
                                     preprocess=preprocess)
    dataloader_img_val = DataLoader(dataset=dataset_img_val,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    collate_fn=my_collate_fn_img)
    dataset_text_val = TaoBao2022_Text(args=args,
                                       img_dict=img_dir_dict,
                                       alt_labels=alt_label_dict,
                                       gt_labels=gt_label_dict,
                                       mode='train',
                                       max_length=20)
    dataloader_text_val = DataLoader(dataset=dataset_text_val,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=my_collate_fn_text)
    val_loader = (dataloader_img_val, dataloader_text_val)

    return train_loader, val_loader

def train_epoch(model, dataloader, optimizers, criterion, metric):
    metric.init()
    model = model.train()
    train_img, train_text = dataloader
    train_loss = 0
    train_log = {'n_images': 0,
                 'acc': 0,
                 'em': 0}

    for idx, (imgs, texts) in enumerate(zip(train_img, train_text)):
        print('training: %d/%d' % (idx, 17273))

        for optimizer in optimizers:
            optimizer.zero_grad()

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        alt_tokens, _, _, gt_labels = texts
        imgs, alt_tokens = imgs.to(device), alt_tokens.to(device).long()
        gt_labels = gt_labels.to(device).long()

        logits = model(imgs, alt_tokens)
        loss = criterion(logits.view(gt_labels.shape[0], -1), gt_labels.view(-1,))
        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        pred_labels = logits.detach().softmax(dim=1).argmax(dim=1)
        train_loss += loss.item()
        metric.update(gt_labels.detach().view(-1,).cpu().tolist(), pred_labels.view(-1,).tolist())
        metric.print_value()
        train_log['n_images'], (train_log['acc'], train_log['em']) = metric.n_images, metric.get_value()
    
    train_loss = train_loss / train_log['n_images']
    train_acc, train_em = train_log['acc'], train_log['em']
    return train_loss, train_acc, train_em

def validation(model, val_loader, criterion, metric):
    metric.init()
    model = model.eval()
    val_loss = 0

    for idx, (imgs, texts) in enumerate(val_loader):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        alt_tokens, _, _, gt_labels = texts
        imgs, alt_tokens = imgs.to(device), alt_tokens.to(device)

        logits = model(imgs, alt_tokens)
        loss = criterion(logits, gt_labels.squeeze())

        pred_labels = logits.detach().softmax(dim=1).argmax(dim=1)
        val_loss += loss.item()
        metric.update(gt_labels.squeeze().tolist(), pred_labels.squeeze().tolist())
        metric.print_value()
    
    val_loss /= metric.n_images
    val_acc, val_em = metric.get_value()
    return val_loss, val_acc, val_em

"""
    Training The Handcraft Model
"""
def train_model(args=None):
    # load model: 1 HandCraft Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultiModalClassifier(feat_dims=args.feat_dims)
    model.load_pretrained_weights()
    model.img_backbone = model.img_backbone.to(device)
    model.text_backbone.backbone = model.text_backbone.backbone.to(device)
    model.text_backbone.out_layers = model.text_backbone.out_layers.to(device)
    model.match_net = model.match_net.to(device)

    # load datasets
    (train_img, train_text), (val_img, val_text) = get_dataloaders(args=args)

    # optimizer
    optimizer_img = torch.optim.SGD(model.img_backbone.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    optimizer_text = torch.optim.SGD(model.text_backbone.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizer_match = torch.optim.SGD(model.match_net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # optimizers = [optimizer_img, optimizer_text, optimizer_match]
    optimizers = [optimizer_img, optimizer_text]

    # scheduler
    scheduler_img = torch.optim.lr_scheduler.StepLR(optimizer_img, step_size=5, gamma=0.5, last_epoch=-1, verbose=False)
    scheduler_text = torch.optim.lr_scheduler.StepLR(optimizer_text, step_size=5, gamma=0.5, last_epoch=-1, verbose=False)
    # scheduler_match = torch.optim.lr_scheduler.StepLR(optimizer_match, step_size=5, gamma=0.5, last_epoch=-1, verbose=False)
    # schedulers = [scheduler_img, scheduler_text, scheduler_match]
    schedulers = [scheduler_img, scheduler_text]

    # criterion
    criterion = nn.CrossEntropyLoss()

    # metric
    metric = TaoBaoMetric()

    # training and validating
    for n_epoch in range(args.num_epochs):
        train_loss, train_acc, train_em = train_epoch(model=model, dataloader=(train_img, train_text), optimizers=optimizers, criterion=criterion, metric=metric)
        # val_acc, val_em = validation(model=model, val_loader=val_loader, criterion=criterion, metric=metric)
        for scheduler in schedulers:
            scheduler.step()

"""
    Finetuning the CLIP Model
"""
def train_epoch_clip_version(epoch, model, dataloader, optimizer, scheduler, criterion, metric, writer: SummaryWriter):
    global train_iters

    metric_epoch, metric_hundred = metric
    metric_epoch.init(), metric_hundred.init()
    model = model.train()

    train_img, train_text = dataloader
    criterion_img, criterion_text = criterion

    train_loss = 0
    train_log = {'n_images': 0,
                 'acc': 0,
                 'em': 0}

    # initialize progress bar
    assert len(train_img) == len(train_text)
    with tqdm(total=len(train_img)) as pbar:
        for idx, (imgs, texts) in enumerate(zip(train_img, train_text)):
            optimizer.zero_grad()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            alt_tokens, _, _, gt_labels = texts
            """
            alt_tokens_in_batch = torch.zeros(0, alt_tokens.shape[-1])
            for label in gt_labels:
                alt_tokens_in_batch = torch.cat((alt_tokens_in_batch, alt_tokens[label.cpu().numpy()]), dim=0)
            """
            # select batch_num pairs of image-text for training
            selected_imgs = torch.zeros(0, 3, 224, 224)
            for label in torch.unique(gt_labels):
                indexes = torch.where(gt_labels.view(-1,) == int(label))[0].numpy()
                shuffle(indexes)
                selected_imgs = torch.cat((selected_imgs, imgs[indexes[0]].unsqueeze(0)), dim=0)
            # imgs, alt_tokens = imgs.to(device), alt_tokens.to(device).long()
            imgs, alt_tokens = selected_imgs.to(device), alt_tokens.to(device).long()
            # batch_size = imgs.shape[0]
            batch_size = alt_tokens.shape[0]

            """
            gt_labels_image = gt_labels.long().to(device)
            class_num = alt_tokens.shape[0]
            gt_labels_text = torch.zeros(batch_size, class_num).to(device)
            gt_labels_text = gt_labels_text.scatter(1, gt_labels_image, 1).T
            """
            gt_labels_image = torch.arange(batch_size).long().to(device)
            gt_labels_text = torch.arange(batch_size).long().to(device)
            
            logits_per_image, logits_per_text = model(imgs, alt_tokens)
            """
            loss_img = criterion_img(logits_per_image, gt_labels_image.view(batch_size,))
            loss_text= criterion_text(logits_per_text, gt_labels_text.view(class_num, batch_size))
            """
            try:
                loss_img = criterion_img(logits_per_image, gt_labels_image.view(batch_size,))
                loss_text= criterion_text(logits_per_text, gt_labels_text.view(batch_size,))
            except:
                pdb.set_trace()
            
            total_loss = (loss_img + loss_text) / 2
            total_loss.backward()

            # convert CLIP model to float32
            for param in model.parameters():
                param.data = param.data.float()
                param.grad.data = param.grad.data.float()

            optimizer.step()
            clip.model.convert_weights(model)

            pred_labels = logits_per_image.detach().softmax(dim=1).argmax(dim=1)
            train_loss += total_loss.item()
            metric_epoch.update(gt_labels_image.detach().view(-1,).cpu().tolist(), pred_labels.view(-1,).tolist())
            metric_hundred.update(gt_labels_image.detach().view(-1,).cpu().tolist(), pred_labels.view(-1,).tolist())
            metric_hundred.increase_loss(total_loss.item())
            # metric.print_value()
            train_log['n_images'], (train_log['acc'], train_log['em']) = metric_epoch.n_images, metric_epoch.get_value()

            # output training info to screen and TensorBoard
            if idx % 100 == 0:
                scheduler.step(metric_epoch.get_acc())
            if idx % 200 == 0:
                writer.add_scalar(tag='loss/train',
                                  scalar_value=metric_hundred.get_loss(),
                                  global_step=train_iters)
                writer.add_scalar(tag='acc/train',
                                  scalar_value=metric_hundred.get_acc(),
                                  global_step=train_iters)
                writer.add_scalar(tag='em/train',
                                  scalar_value=metric_hundred.get_em(),
                                  global_step=train_iters)
                train_iters = train_iters + 1
                metric_hundred.init()

            pbar.update(n=1)

            # save finetuned CLIP model
            if epoch == 0:
                if idx > 0 and idx % 4000 == 0:
                    torch.save(model, 'E:\\DataSets\\PRML2022\\saved_models\\epoch_%d_iter_%d.pkl' % (epoch, idx))
    
    if 0 < epoch <= 5:
        torch.save(model, 'E:\\DataSets\\PRML2022\\saved_models\\epoch_%d' % epoch)
    elif epoch % 5 == 0:
        torch.save(model, 'E:\\DataSets\\PRML2022\\saved_models\\epoch_%d' % epoch)

    train_loss = train_loss / train_log['n_images']
    train_acc, train_em = train_log['acc'], train_log['em']
    return train_loss, train_acc, train_em

def validation_clip_version(model, dataloader, criterion, metric, writer: SummaryWriter):
    global eval_iters
    metric.init()
    model = model.eval()

    val_img, val_text = dataloader

    val_loss = 0
    val_log = {'n_images': 0,
               'acc': 0,
               'em': 0}

    # initialize progress bar
    assert len(val_img) == len(val_text)
    with tqdm(total=len(val_img)) as pbar:
        with torch.no_grad():
            for idx, (imgs, texts) in enumerate(zip(val_img, val_text)):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                alt_tokens, _, _, gt_labels = texts
                imgs, alt_tokens = imgs.to(device), alt_tokens.to(device).long()
                gt_labels = gt_labels.long().to(device)
                batch_size = imgs.shape[0]

                logits_per_image, _ = model(imgs, alt_tokens)
                loss_img = criterion(logits_per_image, gt_labels.view(batch_size,))

                pred_labels = logits_per_image.detach().softmax(dim=1).argmax(dim=1)
                val_loss += loss_img.item()
                metric.update(gt_labels.detach().view(-1,).cpu().tolist(), pred_labels.view(-1,).tolist())
                val_log['n_images'], (val_log['acc'], val_log['em']) = metric.n_images, metric.get_value()

                if idx % 100 == 0:
                    writer.add_scalar(tag='loss/eval',
                                      scalar_value=val_loss / val_log['n_images'],
                                      global_step=eval_iters)
                    writer.add_scalar(tag='acc/eval',
                                      scalar_value=val_log['acc'],
                                      global_step=eval_iters)
                    writer.add_scalar(tag='em/eval',
                                      scalar_value=val_log['em'],
                                      global_step=eval_iters)
                    eval_iters = eval_iters + 1

                # update progress bar
                pbar.update(n=1)
    
    val_loss /= val_log['n_images']
    val_acc, val_em = metric.get_value()
    return val_loss, val_acc, val_em

def train_model_clip_version(args=None):
    # load SummaryWriter from Tensorboard
    writer = SummaryWriter(log_dir="runs/" + "%s" % time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), flush_secs=60)
    # load model: 2 Clip Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load("ViT-B/32", 
                                  device=device, 
                                  download_root='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\clip_models')

    # load datasets
    (train_img, train_text), (val_img, val_text) = get_dataloaders(args=args, preprocess=preprocess)

    # optimizer
    # optimizer_clip = torch.optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    optimizer_clip = torch.optim.Adam(model.parameters(), lr=6e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

    # scheduler
    # scheduler_clip = torch.optim.lr_scheduler.StepLR(optimizer_clip, step_size=10, gamma=0.5, last_epoch=-1, verbose=False)
    # scheduler_clip = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_clip, factor=0.5, mode='max', patience=5, verbose=1, min_lr=1e-9)
    scheduler_clip = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_clip, factor=0.8, mode='max', patience=10, verbose=1, min_lr=5e-9)

    # criterion
    criterion_img = nn.CrossEntropyLoss().to(device)
    criterion_text = nn.CrossEntropyLoss().to(device)
    # criterion_text = nn.BCEWithLogitsLoss().to(device)

    # metric
    metric_epoch, metric_hundred = TaoBaoMetric(), TaoBaoMetric()

    # training and validating
    for n_epoch in range(args.num_epochs):
        train_loss, train_acc, train_em = train_epoch_clip_version(epoch=n_epoch,
                                                                   model=model,
                                                                   dataloader=(train_img, train_text),
                                                                   optimizer=optimizer_clip,
                                                                   scheduler=scheduler_clip,
                                                                   criterion=(criterion_img, criterion_text),
                                                                   metric=(metric_epoch, metric_hundred),
                                                                   writer=writer)
        val_loss, val_acc, val_em = validation_clip_version(model=model,
                                                            dataloader=(val_img, val_text),
                                                            criterion=criterion_img,
                                                            metric=metric_epoch,
                                                            writer=writer)
        # scheduler_clip.step()

    # close Tensorboard SummaryWriter
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train validation split
    parser.add_argument('--train_size', type=int, default=21597)
    # dataloader
    parser.add_argument('--filedir', type=str, default='E:\\DataSets\\PRML2022\\medium\\medium')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--for_clip', default=True, action='store_true')
    parser.add_argument('--use_mask', default=False, action='store_true')
    # model
    parser.add_argument('--feat_dims', type=int, default=512)
    # training
    parser.add_argument('--num_epochs', type=int, default=50)
    
    args = parser.parse_args()

    train_iters, eval_iters = 0, 0

    # train_model(args=args)
    train_model_clip_version(args=args)
