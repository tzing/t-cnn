import torch
import torch.nn as nn
import torch.utils.data as data

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import argparse
import os
from collections import deque
from time import time

import numpy
from PIL import ImageDraw

from data import Folder
from sample import SimpleSampler
from model import Net
from bbreg import BBRegressor


"""
Arg Parse
"""
if __name__ != '__main__':
    exit(1)

psr = argparse.ArgumentParser()
psr.add_argument('dataset')
psr.add_argument('dest')

args = psr.parse_args()
dataset_folder = args.dataset
output_dest = args.dest

# ensure dir exist
output_dest = os.path.abspath(os.path.expanduser(output_dest))
if not os.path.exists(output_dest):
    os.makedirs(output_dest)
elif not os.path.isdir(output_dest):
    print('Not a directory: {output_dest}')
    exit(2)


"""
Constants
"""
CORP_SIZE   = 75

FEATURE_ARCH    = 'vgg11_bn'

SAMPLING_SIGMA_XY   = .3
SAMPLING_SIGMA_S    = .5

FINETUNE_POS_NUM    = 50
FINETUNE_NEG_NUM    = 200
FINETUNE_IOU_THRESHOLD = [.7, .5]

INIT_EPOCH          = 50
INIT_LR             = 1E-3
FINETUNE_EPOCH      = 10
FINETUNE_LR         = 3E-3
FINETUNE_MOMENTUM   = .9
FINETUNE_WDECAY     = 5E-4

MAX_NODE_NUM    = 10
BLOCK_FRAME     = 10

NUM_PREDICT_PATCH   = 256
NUM_PREDICT_USE     = 5


"""
Data loading
"""
print('==> loading data')

folder = Folder(dataset_folder)
print(f'Done. got {len(folder)} frames')

folder = iter(folder)
train_img, train_bbox = next(folder)


"""
dependency
"""
print('==> loading dependency')

# pretrained model as feature extractor
featNet = torchvision.models.__dict__[FEATURE_ARCH](pretrained=True)
featNet = featNet.features

featNet.train(False)

for p in featNet.parameters():
    p.requires_grad = False

featNet = featNet.cuda()
print(f"Model '{FEATURE_ARCH}' loaded")

# image transformer
img_transforms = transforms.Compose([
    transforms.Resize((CORP_SIZE, CORP_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                           std = [ 0.229, 0.224, 0.225 ])
])
print('image transformer loaded')

# sampler
train_sampler = SimpleSampler(
    train_bbox,
    img_transforms,
    target_transforms_pos=lambda x: 0,
    target_transforms_neg=lambda x: 1,
    num=[FINETUNE_POS_NUM, FINETUNE_NEG_NUM],
    threshold=FINETUNE_IOU_THRESHOLD,
    sigma_xy=[SAMPLING_SIGMA_XY, SAMPLING_SIGMA_XY],
    sigma_s=SAMPLING_SIGMA_S
)

test_sampler = SimpleSampler(
    train_bbox,
    img_transforms,
    num=[NUM_PREDICT_PATCH, 0],
    threshold=[0, 1],
    sigma_xy=[SAMPLING_SIGMA_XY, SAMPLING_SIGMA_XY],
    sigma_s=SAMPLING_SIGMA_S
)

# nodes
nodes = deque(maxlen=MAX_NODE_NUM)


"""
training
"""
# finetune on initial frame
print('==> initial finetune')

class Node(object):
    """
    Represent `the CNN` node in the paper
    """

    def __init__(self, regions, epoch=FINETUNE_EPOCH, lr=FINETUNE_LR):
        """
        Perform finetune
        """
        dataset = data.ConcatDataset(regions)
        loader = data.DataLoader(dataset, FINETUNE_POS_NUM+FINETUNE_NEG_NUM)

        net = Net().cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=FINETUNE_MOMENTUM,
            weight_decay=FINETUNE_WDECAY
        )

        tic_train = time()
        for ep in range(epoch):
            tic_epoch = time()
            for imgs, labels in loader:
                imgs = Variable(imgs).cuda()
                labels = Variable(labels).cuda()

                imgs = featNet(imgs)

                optimizer.zero_grad()
                output = net(imgs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

            print(
                f'Epoch[{ep+1}/{epoch}]',
                f'loss={loss.data[0]:.4f}',
                f'time={time()-tic_epoch:.03}s',
                sep='\t'
            )

        print(f'Training finish. Elasped time is {time()-tic_train:.3}sec')
        net.train(False)

        self.net = net
        self.confidence = 1

    def __call__(self, feat):
        """
        Bounding box predicting
        """
        x = self.net(feat)
        x = nn.functional.softmax(x, 1)

        scores, idx = x[:, 0].topk(NUM_PREDICT_USE)
        return scores, idx

node = Node([train_sampler(train_img, train_bbox)], INIT_EPOCH, INIT_LR)
nodes.append(node)

# train bb regression
print('==> bb regression')

dataset = SimpleSampler(
    train_bbox,
    img_transforms,
    num=[FINETUNE_POS_NUM, 0],
    threshold=FINETUNE_IOU_THRESHOLD
)(train_img, train_bbox)
loader = data.DataLoader(dataset, len(dataset))

imgs, boxes = next(iter(loader))

feat = Variable(imgs).cuda()
feat = featNet(feat)

bbreg = BBRegressor()
bbreg.fit(feat, boxes, train_bbox)

print('done')


"""
predicting
"""
print('==> predicting')

recent_scores = []
recent_regions = []

last_bbox = train_bbox

for frame_idx, (frame_img, _) in enumerate(folder):
    print(f'Frame[{frame_idx}]')

    dataset = test_sampler(frame_img, last_bbox)
    loader = data.DataLoader(dataset, len(dataset))

    # first attempt
    imgs, boxs = next(iter(loader))

    feat = Variable(imgs).cuda()
    feat = featNet(feat)

    # predict use all candidates
    weight = torch.FloatTensor(len(nodes))
    scores = torch.FloatTensor(len(nodes), NUM_PREDICT_USE)
    indices = torch.LongTensor(len(nodes), NUM_PREDICT_USE)

    for i, node in enumerate(nodes):
        score, idxs = node(feat)
        weight[i] = node.confidence
        scores[i] = score.data
        indices[i] = idxs.data

    # modify weight ...see eq(6)
    weight /= torch.sum(weight)
    for i, w in enumerate(weight):
        scores[i] *= w

    # flatten
    scores = scores.view(-1)
    indices = indices.view(-1)

    # select sample with max modified score
    conf, idx = scores.topk(NUM_PREDICT_USE)

    # make predict
    predict = torch.mean(boxs[indices[idx], :], 0).numpy()
    confidence = torch.mean(conf)

    # bb regression
    img = frame_img.crop(tuple(predict))
    img = img_transforms(img)
    img.unsqueeze_(0)

    img = Variable(img).cuda()
    img = featNet(img)

    predict = bbreg.predict(img, predict)
    predict = predict.flatten()

    # draw
    out_img = frame_img.copy()

    draw = ImageDraw.Draw(out_img)
    draw.rectangle(tuple(predict), outline='red')
    del draw

    # output
    print(predict, f'conf={confidence:.4f}')

    fn = os.path.join(output_dest, f'{frame_idx:06}.jpg')
    out_img.save(fn)

    # save
    last_bbox = predict

    recent_scores.append(confidence)
    recent_regions.append(train_sampler(frame_img, predict))

    # continue predicting
    if len(recent_scores) < BLOCK_FRAME:
        continue

    # train new model
    print('>> train new finetune model')
    node = Node(recent_regions)

    beta_conf = numpy.mean(recent_scores)
    all_confs = [n.confidence for n in nodes]
    all_confs = numpy.minimum(all_confs, beta_conf)

    node.confidence = numpy.max(all_confs)
    nodes.append(node)

    recent_scores = []
    recent_regions = []
