from sklearn.linear_model import Ridge

import torch
import numpy

def ensure_numpy(x):
    if isinstance(x, numpy.ndarray):
        return x
    elif torch.is_tensor(x):
        return x.cpu().numpy()
    elif isinstance(x, torch.autograd.Variable):
        return ensure_numpy(x.data)
    raise TypeError('Unsupported type of input')

def ensure_shape(x):
    if len(x.shape) == 2:
        return x
    if len(x.shape) == 1:
        return x.reshape((1, -1))
    return x.reshape((x.shape[0], -1))

def check_input(feat, bbox):
    feat = ensure_shape(ensure_numpy(feat))
    bbox = ensure_shape(ensure_numpy(bbox))
    assert feat.shape[0] == bbox.shape[0]
    return feat, bbox

def chRep_pts2xywh(box):
    """
    change representation

    from: (left-top, bottom-right)
    to:   (center, size)
    """
    lt = box[:, :2]
    br = box[:, 2:]

    ctr = (br + lt) /2
    size = br - lt
    return ctr, size

def chRep_xywh2pts(xy, wh):
    """
    change representation

    from: (center, size)
    to:   (left-top, bottom-right)
    """
    wh /= 2
    lt = xy - wh
    br = xy + wh
    return numpy.concatenate([lt, br])

class BBRegressor(Ridge):
    """
    The bounding-box regessior
    """

    def fit(self, feat, bbox, gt):
        feat, bbox = check_input(feat, bbox)
        gt = ensure_shape(ensure_numpy(gt))

        # check size
        assert len(gt.shape)==2 and gt.shape[0]==1
        n = feat.shape[0]

        # change representation
        bbox_ctr, bbox_size = chRep_pts2xywh(bbox)
        gt_ctr, gt_size = chRep_pts2xywh(gt)

        # tile
        gt_ctr = numpy.tile(gt_ctr, (n, 1))
        gt_size = numpy.tile(gt_size, (n, 1))

        # generate internal predict target
        Yxy = (gt_ctr - bbox_ctr) / bbox_size
        Ywh = numpy.log(gt_size / bbox_size)

        Y = numpy.concatenate([Yxy, Ywh], axis=1)

        # fitting internal target
        super().fit(feat, Y)

    def predict(self, feat, bbox):
        feat, bbox = check_input(feat, bbox)
        bbox_ctr, bbox_size = chRep_pts2xywh(bbox)

        # predict internal target "Y"
        Y = super().predict(feat)
        Yxy, Ywh = Y[:, :2], Y[:, 2:]

        # get final target "T"
        Txy = bbox_ctr + Yxy * bbox_size
        Twh = numpy.exp(Ywh) * bbox_size

        return chRep_xywh2pts(Txy, Twh)
