import numpy as np
import torch
from torchvision import transforms
import torch.nn.functional as F
import clip
import argparse
from preprocessing import load_data, load_data_pure_feature
from criterion import TaoBaoMetric
from category import cluster_classification
from tqdm import tqdm

import pdb
import os
import json


# clip inference
def clip_inference_main(args=None):
    # declarations
    data_dir = 'E:\\DataSets\\PRML2022\\medium\\medium'
    model_dir = 'E:\\DataSets\\PRML2022\\saved_models\\epoch_0_iter_4000.pkl'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # models and datasets
    _, preprocess = clip.load("ViT-B/32", device=device, download_root='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\clip_models')  # default:'~/.cache/clip'
    model = torch.load(model_dir)
    _, test_loader = load_data(filedir=data_dir, args=args)
    Tensor2Image = transforms.ToPILImage()

    # metric
    metric = TaoBaoMetric()
    metric.init()

    """
    # inference (training set)
    for idx, (imgs, alt_texts_, gt_texts) in enumerate(train_loader):
        print(idx)
        alt_texts = [text[0].lower() for text in alt_texts_]
        clip_texts = [text[0].lower() + ' clothes' for text in alt_texts_]
        gt_texts = [text[0].lower() for text in gt_texts]
        text = clip.tokenize(clip_texts).to(device)

        batch_size = len(imgs)
        pred_labels = []
            
        # text_features = F.normalize(text_features, dim=1)

        # inference a single batch

        # image_features = torch.zeros(batch_size, 2048)

        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features_all = np.zeros((0, 512))
            for batch_idx in range(batch_size):
                image = Tensor2Image(imgs[batch_idx].squeeze().permute(2, 0, 1))
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = model.encode_image(image)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features_all = np.concatenate((image_features_all, image_features.cpu().numpy()), axis=0)
                # Method1: distance metric
                # dist_features = 1 / (1 + np.linalg.norm(image_features.cpu().unsqueeze(1) - text_features.cpu().unsqueeze(0), axis=-1))

                # Method2: similariy metric
                # similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                # _, indices = similarity[0].topk(1)
                # pred_labels.append(alt_texts[int(indices.cpu().numpy())])

                # Other Methods:
                # image_features = F.normalize(image_features, dim=1).view(1, -1)
                # calculate pair-to-pair similarity
                # similarity = torch.matmul(image_features, text_features.T).cpu().numpy()
                # logits_per_image, logits_per_text = model(image, text)
                # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                
                # image_features[batch_idx, ...] = model.encode_image(image)
            # Method3: cluster before classification
            pred_labels = cluster_classification(imgs_feats=image_features_all, 
                                                 text_feats=text_features.cpu().numpy(),
                                                 opt_tags=alt_texts)
            pred_labels = [alt_texts[idx] for idx in pred_labels]

        print(gt_texts, pred_labels)
        metric.update(gt_texts, pred_labels)
        metric.print_value()
    """

    # inference (testing set)
    pred_file = open(os.path.join(data_dir, 'test_all.json'), 'r', encoding='utf-8')
    pred_data = json.load(pred_file)
    pred_file.close()

    with tqdm(total=len(test_loader)) as pbar:
        for idx, (imgs, alt_texts_) in enumerate(test_loader):
            alt_texts = [text[0] for text in alt_texts_]
            # clip_texts = [text[0].lower() + ' clothes' for text in alt_texts_]
            alt_texts_cn = pred_data[list(pred_data.keys())[idx]]['optional_tags']
            text = clip.tokenize(alt_texts).to(device)
            batch_size = len(imgs)
            pred_indices = []

            with torch.no_grad():
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                image_features_all = np.zeros((0, 512))
                for batch_idx in range(batch_size):
                    image = Tensor2Image(imgs[batch_idx].squeeze().permute(2, 0, 1))
                    image = preprocess(image).unsqueeze(0).to(device)
                    image_features = model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    image_features_all = np.concatenate((image_features_all, image_features.cpu().numpy()), axis=0)
                    
                    # calculate pair-to-pair similarity

                    """
                        @ Method 1:
                          Decide using the largest logits
                    """
                    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                    _, indices = similarity[0].topk(1)
                    
                    key = list(pred_data[list(pred_data.keys())[idx]]['imgs_tags'][batch_idx].keys())[0]
                    pred_data[list(pred_data.keys())[idx]]['imgs_tags'][batch_idx][key] = alt_texts_cn[indices]
                    pred_indices.append(indices)

                # @ Method2
                if len(np.unique(np.array(pred_indices)).tolist()) < len(alt_texts):
                    print('launching cluster classification')
                    indices = cluster_classification(imgs_feats=image_features_all, 
                                                     text_feats=text_features.cpu().numpy(),
                                                     opt_tags=alt_texts,
                                                     ret_indices=True)

                    for batch_idx in range(batch_size):
                        key = list(pred_data[list(pred_data.keys())[idx]]['imgs_tags'][batch_idx].keys())[0]
                        pred_data[list(pred_data.keys())[idx]]['imgs_tags'][batch_idx][key] = alt_texts_cn[indices[batch_idx]]
            pbar.update(n=1)

    pred_file = open(os.path.join(data_dir, 'test_all_pred.json'), 'w', encoding='utf-8')
    json.dump(pred_data, pred_file, indent=2, ensure_ascii=False)

# inference with pure image features
def clip_feature_inference_main(args=None):
    data_dir = 'E:\\DataSets\\PRML2022\\medium\\medium'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, preprocess = clip.load("ViT-L/14@336px", device=device, download_root='D:\\Tsinghua\\bachelor\\sem3_2\\prml\\clip_models')  # default:'~/.cache/clip'
    test_loader = load_data_pure_feature(filedir=data_dir, args=args)

    for idx, (imgs_feats, texts_feats) in enumerate(test_loader):
        print(idx)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1,
                        help='number of sample images in a minibatch')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num of threads for loading dataset')
    args = parser.parse_args()
    clip_inference_main(args=args)
