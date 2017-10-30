from torch.utils import data

import numpy


COEF_SCALE = 1.05


class RegionDataset(data.Dataset):

    def __init__(self,
                 img,
                 crops,
                 transforms=None,
                 target_transforms_pos=None,
                 target_transforms_neg=None
                ):
        self.img = img
        self.crops = crops
        self.transforms = transforms
        self.target_transforms_pos = target_transforms_pos
        self.target_transforms_neg = target_transforms_neg

    def __len__(self):
        return len(self.crops)

    def __getitem__(self, index):
        is_pos, bbox = self.crops[index]

        img = self.img.crop(tuple(bbox))
        target = bbox

        if self.transforms is not None:
            img = self.transforms(img)

        if is_pos and (self.target_transforms_pos is not None):
            target = self.target_transforms_pos(target)
        elif (not is_pos) and (self.target_transforms_neg is not None):
            target = self.target_transforms_neg(target)

        return img, target


class SimpleSampler(object):
    """
    This class implement the sampler in the paper
    """

    def __init__(self,
                 init_bbox,                 # bbox(bounding box) of initial frame
                 transforms=None,           # image transforms (from PIL)
                 target_transforms_pos=None,# positive target transforms (from bbox)
                 target_transforms_neg=None,# negative target transforms (from bbox)
                 num=[50, 50],              # (pos, neg) num of samples to return
                 threshold=[.7, .5],        # (pos, neg) threshold IoU of samples
                 sigma_xy=[.3, .3],         # (x, y) sigma of translation
                 sigma_s=.5                 # sigma of scaling
                ):

        ibox = numpy.array(init_bbox)
        lt, br = ibox.reshape([2, 2])
        self.isize = br - lt

        self.transforms = transforms
        self.target_transforms_pos = target_transforms_pos
        self.target_transforms_neg = target_transforms_neg
        self.num = numpy.array(num)
        self.num_total = sum(num)
        self.thres_pos, self.thres_neg = threshold
        self.sigma_t = numpy.array(sigma_xy)
        self.sigma_s = sigma_s

    def __call__(self, img, bbox):
        bbox = numpy.array(bbox)
        lt, br = bbox.reshape([2, 2])

        ctr = (lt + br) /2
        sz = numpy.mean(br - lt)
        area = numpy.prod(sz)

        sigma = self.sigma_t * sz

        data = []
        remain = self.num.copy()
        def add_data(is_pos, nlt, nbr):
            idx = int(not is_pos)
            if remain[idx] == 0:
                return

            remain[idx] -= 1
            data.append((
                is_pos,
                numpy.concatenate([nlt, nbr])
            ))

        while True:
            # gen region
            nctr = numpy.random.normal(ctr, sigma)
            nsize_2 = self.isize * (COEF_SCALE ** numpy.random.normal(scale=self.sigma_s)) /2
            nlt = nctr - nsize_2
            nbr = nctr + nsize_2

            # overlap test
            if (br < nlt).any() or (lt > nbr).any():
                add_data(False, nlt, nbr)
                continue

            # iou
            olt = numpy.maximum(lt, nlt)
            obr = numpy.minimum(br, nbr)

            area_over = numpy.prod(obr-olt)
            area_add = numpy.prod(nsize_2) *4
            area_union = area + area_add - area_over

            iou = area_over / area_union

            # iou test
            if iou >= self.thres_pos:
                add_data(True, nlt, nbr)
            elif iou < self.thres_neg:
                add_data(False, nlt, nbr)

            # loop breaker
            if all(remain==0):
                break

        return RegionDataset(
            img,
            data,
            self.transforms,
            self.target_transforms_pos,
            self.target_transforms_neg
        )
