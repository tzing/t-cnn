import torch
import torch.nn as nn
import torch.utils.data as data

from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

from data import Folder
from sample import SimpleSampler
from model import Net
from bbreg import BBRegressor


"""
Constants
"""
FOLDER      = '~/Dataset/vot/bag'
CORP_SIZE   = 75

FEATURE_ARCH    = 'vgg11_bn'

FINETUNE_POS_NUM    = 50
FINETUNE_NEG_NUM    = 200
FINETUNE_IOU_THRESHOLD = [.7, .5]

INIT_EPOCH          = 50
INIT_LR             = 1E-3
FINETUNE_EPOCH      = 10
FINETUNE_LR         = 3E-3
FINETUNE_MOMENTUM   = .9
FINETUNE_WDECAY     = 5E-4

BLOCK_FRAME     = 10

NUM_PREDICT_PATCH   = 256
NUM_PREDICT_USE     = 5


"""
Data loading
"""
print('==> loading data')

folder = Folder(FOLDER)
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
    threshold=FINETUNE_IOU_THRESHOLD
)

test_sampler = SimpleSampler(
    train_bbox,
    img_transforms,
    num=[NUM_PREDICT_PATCH, 0],
    threshold=[0, 1]
)


"""
training
"""
# finetune on initial frame
print('==> initial finetune')

def finetune(regions, epoch=FINETUNE_EPOCH, lr=FINETUNE_LR):
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

    for ep in range(epoch):
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
            sep='\t'
        )

    return net

net = finetune([train_sampler(train_img, train_bbox)], INIT_EPOCH, INIT_LR)

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
net.train(False)

for frame_idx, (frame_img, _) in enumerate(folder):
    print(f'Frame[{frame_idx}]')

    dataset = test_sampler(frame_img, last_bbox)
    loader = data.DataLoader(dataset, len(dataset))

    # first attempt
    imgs, boxs = next(iter(loader))

    x = Variable(imgs).cuda()
    x = featNet(x)
    x = net(x)

    x = nn.functional.softmax(x, 1)

    score, idx = x[:, 0].topk(NUM_PREDICT_USE)

    score = torch.mean(score.data)
    idx = idx.data.cpu()

    predict = torch.mean(boxs[idx, :], 0).numpy()

    # bb regression
    img = frame_img.crop(tuple(predict))
    img = img_transforms(img)
    img.unsqueeze_(0)

    img = Variable(img).cuda()
    img = featNet(img)

    predict = bbreg.predict(img, predict)
    predict = predict.flatten()

    # print
    print(predict, f'conf={score:.4f}')

    # save
    last_bbox = predict

    recent_scores.append(score)
    recent_regions.append(train_sampler(frame_img, predict))

    # continue predicting
    if len(recent_scores) < BLOCK_FRAME:
        continue

    # train new model
    print('>> train new finetune model')
    net = finetune(recent_regions)
    break
