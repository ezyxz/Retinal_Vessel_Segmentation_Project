from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class VOCSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=None,
                 split='train',
                 transform=None #composed_transforms_tr
                 ):

        self._base_dir = base_dir  # './data_slice_{}'.format('CHASE')
        self.split = split         # 'train'
        self._image_dir = os.path.join(self._base_dir, 'image_'+self.split)  #'./data_slice_CHASE/image_train'
        self._cat_dir = os.path.join(self._base_dir, 'label_'+self.split)    #'./data_slice_CHASE/label_train'
        self.images = os.listdir(self._image_dir)
        self.categories = []
        for i in range(len(self.images)):
            self.categories.append(self.images[i])

        self.transform = transform


        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        _img = np.asarray(Image.open(os.path.join(self._image_dir, self.images[index])).convert('RGB')).astype(np.float32)
        _target = np.asarray(Image.open(os.path.join(self._cat_dir, self.categories[index])).convert('L')).astype(np.int32)
        # _img = io.read_image(os.path.join(self._image_dir, self.images[index]), driver = 'GDAL')
        # _target = np.asarray(Image.open(os.path.join(self._cat_dir, self.categories[index])).convert('L')).astype(
        #     np.int32)
        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'

