# T-CNN

pytorch implement of *"Modeling and propagating cnns in a tree structure for visual tracking"*

original paper:

```bibtex
@article{nam2016modeling,
  title={Modeling and propagating cnns in a tree structure for visual tracking},
  author={Nam, Hyeonseob and Baek, Mooyeol and Han, Bohyung},
  journal={arXiv preprint arXiv:1608.07242},
  year={2016}
}
```

## requirements

* python 3.6+
* pytorch
* torchvision
* scikit-learn

## execute

```
usage: main.py [-h] dataset dest

positional arguments:
  dataset
  dest
```

this code use [VOT dataset] and will use ground truth of first frame as tracking target

[VOT dataset]: http://www.votchallenge.net/vot2016/dataset.html

## note

in original it use *[VGG-M]* for feature extraction. Since there's no pretrained *VGG-M* model in PyTorch, it use `vgg11_bn` in this code and the output image size could be same in the paper.

[VGG-M]: https://arxiv.org/abs/1405.3531
