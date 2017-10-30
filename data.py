from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader

import os
import numpy


class Folder(object):
    """
    This class handle the image loading and parse groundtruth file
    form the VOT dataset folder
    """

    def __init__(self, folder, loader=default_loader):
        # check path
        folder = os.path.abspath(os.path.expanduser(folder))
        if not os.path.isdir(folder):
            raise ValueError('Not a valid foler: %s' %folder)

        fn_gtruth = os.path.join(folder, 'groundtruth.txt')
        if not os.path.isfile(fn_gtruth):
            raise ValueError('Groundtruth file not found: %s' %fn_gtruth)

        # load file names
        fnames = sorted(
            f for f in os.listdir(folder)
            if any(f.endswith(ext) for ext in IMG_EXTENSIONS)
        )
        gtruth = numpy.genfromtxt(fn_gtruth, delimiter=',')

        # store data
        self.loader = loader
        self.data = []

        for fn, gt in zip(fnames, gtruth):
            # filename prefix
            fn = os.path.join(folder, fn)

            # gen bbox
            pts_x = gt[0::2]
            pts_y = gt[1::2]

            min_x = min(*pts_x)
            min_y = min(*pts_y)
            max_x = max(*pts_x)
            max_y = max(*pts_y)

            bbox = numpy.array((min_x, min_y, max_x, max_y))

            # save
            self.data.append((fn, bbox))

    def __getitem__(self, index):
        fn, gt = self.data[index]
        img = self.loader(fn)
        return img, gt

    def __len__(self):
        return len(self.data)
