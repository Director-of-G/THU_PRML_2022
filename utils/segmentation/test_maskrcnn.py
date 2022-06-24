# import torch
# import torchvision




# def maskrcnn_segmentation_main(args=None):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
#                                                                progress=True,
#                                                                num_classes=91)

from operator import index
import cv2
import numpy as np
import random
import torch
import torchvision
import argparse
from PIL import Image
from torchvision.transforms import transforms as transforms
from torch.utils.data import DataLoader

import sys
import os

sys.path.append('../')
from preprocessing import load_data, load_labels
from dataloader import TaoBao2022

import pdb
import warnings
warnings.filterwarnings('ignore')
coco_names= [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
 
def get_outputs(image, model, threshold):
    with torch.no_grad():
        # forward pass of the image through the modle
        outputs = model(image)
    
    # get all the scores
    scores = list(outputs[0]['scores'].detach().cpu().numpy())
    # index of those scores which are above a certain threshold
    thresholded_preds_inidices = [scores.index(i) for i in scores if i > threshold]
    thresholded_preds_count = len(thresholded_preds_inidices)
    # get the masks
    masks = (outputs[0]['masks']>0.5).squeeze().detach().cpu().numpy()
    # discard masks for objects which are below threshold
    masks = masks[:thresholded_preds_count]
    # get the bounding boxes, in (x1, y1), (x2, y2) format
    boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[0]['boxes'].detach().cpu()]
    # discard bounding boxes below threshold value
    boxes = boxes[:thresholded_preds_count]
    # get the classes labels
    labels = [coco_names[i] for i in outputs[0]['labels']]
    # person index
    idx = labels.index('person')
    np.savez_compressed('./mask', a=masks[idx, :])
    print(labels[:len(masks)])
    return masks, boxes, labels
 
def draw_segmentation_map(image, masks, boxes, labels):
    alpha = 1 
    beta = 0.6 # transparency for the segmentation map
    gamma = 0 # scalar added to each sum
    for i in range(len(masks)):
        red_map = np.zeros_like(masks[i]).astype(np.uint8)
        green_map = np.zeros_like(masks[i]).astype(np.uint8)
        blue_map = np.zeros_like(masks[i]).astype(np.uint8)
        # apply a randon color mask to each object
        color = COLORS[random.randrange(0, len(COLORS))]
        red_map[masks[i] == 1], green_map[masks[i] == 1], blue_map[masks[i] == 1]  = color
        # combine all the masks into a single image
        segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
        #convert the original PIL image into NumPy format
        image = np.array(image)
        # convert from RGN to OpenCV BGR format
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # apply mask on the image
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma, image)
        # draw the bounding boxes around the objects
        cv2.rectangle(image, boxes[i][0], boxes[i][1], color=color, 
                      thickness=2)
        # put the label text above the objects
        cv2.putText(image , labels[i], (boxes[i][0][0], boxes[i][0][1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 
                    thickness=2, lineType=cv2.LINE_AA)
    
    return image

def maskrcnn_segmentation_main(args=None):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
                                                               num_classes=91)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    # img_dir_dict, alt_label_dict, gt_label_dict = load_labels(filedir=args.filedir, dataset='train')
    img_dir_dict, alt_label_dict = load_labels(filedir=args.filedir, dataset='test')
    train_dataset = TaoBao2022(img_dir_dict, alt_label_dict, datadir=os.path.join(args.filedir, 'test'), dataset='test', ret_Image=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    Tensor2Image = transforms.ToPILImage()
    for shop_idx, (imgs, shop_id) in enumerate(train_loader):
        if shop_idx <= 2622:
            print(shop_idx)
            continue
        for img_idx, img in enumerate(imgs):
            img = Tensor2Image(img.squeeze().permute(2, 0, 1))
            img = img.convert('RGB')
            img = transform(img)
            img = img.unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img)
                masks = (outputs[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
                if len(masks.shape) != 3:
                    masks = np.expand_dims(masks, 0)
                labels = [coco_names[i] for i in outputs[0]['labels']]
                if 'person' not in labels:
                    mask_areas = [np.sum(masks[i]) for i in range(len(masks))]
                    if len(mask_areas) > 0:
                        index = np.argmax(mask_areas)
                        np.savez_compressed(os.path.join(args.filedir, 'test', shop_id[0], 'mask_%d' % img_idx), mask=masks[index])
                    else:
                        np.savez_compressed(os.path.join(args.filedir, 'test', shop_id[0], 'mask_%d' % img_idx), mask=np.ones((img.shape[2], img.shape[3])))
                else:
                    index = labels.index('person')
                    np.savez_compressed(os.path.join(args.filedir, 'test', shop_id[0], 'mask_%d' % img_idx), mask=masks[index])
        print(shop_idx)

 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input',
                    help='path to the input data')
parser.add_argument('-t', '--threshold', default=0.35, type=float,
                    help='score threshold for discarding detection')
parser.add_argument('-f', '--filedir', default='E:\\DataSets\\PRML2022\\medium\\medium', type=str,
                    help='path to the input data')
parser.add_argument('--batch_size', type=int, default=1,
                    help='number of sample images in a minibatch')
parser.add_argument('--num_workers', type=int, default=0,
                    help='num of threads for loading dataset')
args = parser.parse_args()

maskrcnn_segmentation_main(args=args)
 
# # initialize the model
# model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True, 
#                                                            num_classes=91)
# # set the computation device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # load the modle on to the computation device and set to eval mode
# model.to(device).eval()
# # transform to convert the image to tensor
# transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# image_path = args['input']
# image = Image.open(image_path).convert('RGB')
# # keep a copy of the original image for OpenCV functions and applying masks
# orig_image = image.copy()
# # transform the image
# image = transform(image)
# # add a batch dimension
# image = image.unsqueeze(0).to(device)
# masks, boxes, labels = get_outputs(image, model, args['threshold'])
# result = draw_segmentation_map(orig_image, masks, boxes, labels)
# # visualize the image
# cv2.imshow('Segmented image', result)
# cv2.waitKey(0)
# # set the save path
# save_path = f"./result.jpg"
# cv2.imwrite(save_path, result)
