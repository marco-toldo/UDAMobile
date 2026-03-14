"""
Utility functions
"""

import numpy as np
import tensorflow as tf
import random


''' 
Cityscapes and GTA ground truth map:
0 = road -> [128, 64,128]
1 = sidewalk -> [128, 0, 0]   
2 = building -> [70, 70, 70]
3 = wall ->[102,102,156]
4 = fence ->[190,153,153]
5 = pole ->[153,153,153]
6 = traffic light ->[250,170, 30]
7 = traffic sign ->[220,220,  0]
8 = vegetation ->[107,142, 35]
9 = terrain ->[152,251,152]
10 = sky ->[70,130,180]
11 = person ->[220, 20, 60]
12 = rider ->[255,  0,  0]
13 = car ->[0, 0, 142]
14 = truck ->[0,  0, 70]
15 = bus ->[0, 60,100]
16 = train ->[0, 80,100]
17 = motorcycle ->[0,  0,230]
18 = bicycle ->[119, 11, 32]
255 = unlabeled, ego vehicle, rectification border, out of roi, static, dynamic, ground, parking, rail track, 
      guard rail, bridge, tunnel, polegroup, caravan, trailer
'''

''' 
SYNTHIA ground truth map:

1 = sky ->[70,130,180]
2 = building -> [70, 70, 70]
3 = road -> [128, 64,128]
4 = sidewalk -> [128, 0, 0]
5 = fence -> [190,153,153]
6 = vegetation -> [107,142, 35]
7 = pole ->[153,153,153]
8 = car ->[0, 0, 142]
9 = traffic sign ->[220,220,  0]
10 = person ->[220, 20, 60]
11 = bicycle ->[119, 11, 32]
12 = motorcycle ->[0,  0,230]
13 = parking-slot  
14 = road-work
15 = traffic light ->[250,170, 30]
16 = terrain
17 = rider ->[255,  0,  0]
18 = truck
19 = bus ->[0, 60,100]
20 = train
21 = wall ->[102,102,156]
22 = lanemarking


'''



# IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
# DATASET = 'SYNTHIA'  # Allowed value: VOC2012, Cityscapes
# DATASET = 'Cityscapes'


# if train_D.SUB_MEAN:
#    IMG_MEAN_GTA = np.array((112.8916, 111.7989, 108.3913), dtype=np.float32)
#    IMG_MEAN_CS = np.array((73.16099, 82.92411, 72.38631), dtype=np.float32)
# else:
#    IMG_MEAN_GTA = np.array((127.5, 127.5, 127.5), dtype=np.float32)
#    IMG_MEAN_CS = np.array((127.5, 127.5, 127.5), dtype=np.float32)

# IMG_MEAN_GTA = np.array((127.5, 127.5, 127.5), dtype=np.float32)
# IMG_MEAN_CS = np.array((127.5, 127.5, 127.5), dtype=np.float32)



def convert_output2rgb(image, dataset):
    """
    Convert the output tensor of the generator to an RGB image

    :param image: outptut tensor given by the generator. 4D tensor: [batch_size, image_width, image_height, num_classes]
    :param dataset: Cityscapes or SYNTHIA class color scheme
    :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    table = None
    if dataset == 'SYNTHIA':
        table = tf.constant(
            [[128, 64, 128], [128, 0, 0], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 60, 100], [0, 0, 230], [119, 11, 32]], tf.uint8)
    elif dataset == 'Cityscapes':
        table = tf.constant(
            [[128, 64, 128], [128, 0, 0], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]], tf.uint8)
    else:
        raise ValueError('accepted dataset names: SYNTHIA or Cityscapes')

    labels = tf.argmax(image, axis=3)
    out_RGB = tf.nn.embedding_lookup(table, labels)
    return out_RGB


def convert_gt2rgb(image, dataset):
    """
    Convert the tensor containing the GT to an RGB image

    :param image: tensor containing gt labels. 4D tensor: [batch_size, image_width, image_height, 1]
    :param dataset: Cityscapes or SYNTHIA class color scheme
    :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
    """
    table = None
    if dataset == 'SYNTHIA':
        table = tf.constant(
            [[128, 64, 128], [128, 0, 0], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 60, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             # fix wrong classes using the value of the background (Only for better visualization of gt)
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255]], tf.uint8)
    elif dataset == 'Cityscapes':
        table = tf.constant(
            [[128, 64, 128], [128, 0, 0], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
             [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0],
             [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32],
             # fix wrong classes using the value of the background (Only for better visualization of gt)
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255]], tf.uint8)
    else:
        raise ValueError('accepted dataset names: SYNTHIA or Cityscapes')

    labels = tf.squeeze(image, axis=3)
    labels = tf.cast(labels, tf.int32)
    out_RGB = tf.nn.embedding_lookup(table, labels)
    return out_RGB



def convert2int(image, dataset_mean=None):
    """Convert input image from [-1, 1] to [0, 255] value ranges.

        Args:
          image: An 4D or 3D image tensor in [-1, 1] format
          dataset_mean: RGB color mean in [0, 255] format

        Returns:
          Converted image.
        """

    image = tf.cast(image, tf.float32)

    mean_std = np.array((127.5, 127.5, 127.5), dtype=np.float32)

    if dataset_mean is not None:
        img = (image * 127.5 + dataset_mean) / 255.
    else:
        img = (image * 127.5 + mean_std) / 255.

    # Perform a cast to [0,1] by adjusting pixel color components which are >1 or <0
    img = tf.minimum(img, 1.)
    img = tf.maximum(img, 0.)

    return tf.image.convert_image_dtype(img, dtype=tf.uint8)


def convert2mask(image, dataset_mean=None):
    """Find the pixels out of range.

            Args:
              image: An 4D or 3D image tensor in [-1, 1] format
              dataset_mean: RGB color mean in [0, 255] format

            Returns:
              Pixel mask.
            """

    image = tf.cast(image, tf.float32)

    mean_std = np.array((127.5, 127.5, 127.5), dtype=np.float32)

    if dataset_mean is not None:
        img = (image * 127.5 + dataset_mean) / 255.
    else:
        img = (image * 127.5 + mean_std) / 255.

    img_gt1 = tf.maximum(img - 1.0001, 0.)
    img_lt0 = tf.maximum((-1) * img, 0.)
    img = img_gt1 + img_lt0

    try:
        img = tf.reduce_mean(img, 3)  # 4D input
    except:
        img = tf.reduce_mean(img, 2)  # 3D input

    img = tf.minimum(img * 10000, 1)

    return tf.image.convert_image_dtype(img, dtype=tf.uint8)


class ImagePool:
    """ History of generated images
        Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []

    def query(self, image): # if batch_size is 1, then image is a single image
        # if batch_size > 1, then self.images becomes a list of batches, not single images
        if self.pool_size == 0:  # no pool
            return image

        if len(self.images) < self.pool_size:  # pool still not full
            self.images.append(image)
            return image
        else:
            p = random.random()
            if p > 0.5:
                # use old image
                random_id = random.randrange(0, self.pool_size)
                tmp = self.images[random_id].copy()  # copy() -> shallow copy, which constructs a new compound object and then
                # (to the extent possible) inserts references into it to the objects found
                # in the original
                self.images[random_id] = image.copy()
                return tmp
            else:
                return image
