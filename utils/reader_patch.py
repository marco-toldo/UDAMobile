
import tensorflow as tf
import math


class Patch_Reader():

    # Constructor
    def __init__(self, tfrecords_file, width_res, crop_size, min_queue_examples=1000, batch_size=4, num_threads=8, seed=None, name=''):
        """
        Args:
          tfrecords_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """
        self.tfrecords_file = tfrecords_file
        self.width_res = width_res
        self.crop_size = crop_size
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name
        self.seed = seed


    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, image_depth]
        """
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])  # 'tfrecords_file' is the path to the tfrecords file
            reader = tf.TFRecordReader()

            _, serialized_example = self.reader.read(filename_queue)        # the tfrecords file is made of many serialized 'examples' (structure: image name, image file)
            sample = tf.parse_single_example(                             # 'tf.parse_single_example' returns a dict mapping feature keys to Tensor and SparseTensor values.
                serialized_example,
                features={
                    'height': tf.FixedLenFeature([], tf.int64),
                    'width': tf.FixedLenFeature([], tf.int64),
                    'input': tf.FixedLenFeature([], tf.string),
                    'gt': tf.FixedLenFeature([], tf.string)
                })

            height = tf.cast(sample['height'], tf.int32)
            width = tf.cast(sample['width'], tf.int32)
            image = tf.image.decode_png(sample['input'], channels=3)
            gt = tf.image.decode_png(sample['gt'], channels=1)


            ###############
            image, gt = self.preprocess_data(image, gt, height, width, width_res=self.width_res, crop_size = self.crop_size)
            ###############


            input_images, gt_images = tf.train.shuffle_batch(
                [image, gt], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3*self.batch_size,
                min_after_dequeue=self.min_queue_examples, seed=self.seed
            )

        return input_images, gt_images


    def preprocess_data(self, image, gt, height, width, width_res, crop_size):

        image = tf.cast(image, tf.float32)
        image = (2./255.)*image - 1.

        width_res_tmp = tf.math.ceil( tf.cast(crop_size[0],tf.float32) * tf.cast(width,tf.float32) / tf.cast(height,tf.float32) )
        width_res = tf.math.maximum( tf.cast(width_res, tf.int32),  tf.cast(width_res_tmp, tf.int32))

        try:
            height_res = tf.cast(tf.cast(height, tf.float32)/tf.cast(width, tf.float32)*tf.cast(width_res, tf.float32), tf.int32)
            image_res = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [height_res, width_res]), axis=0)
            gt_res = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(gt, 0), [height_res, width_res]), axis=0)

            gt_res = tf.cast(gt_res, tf.float32)

            combined = tf.concat([image_res, gt_res], axis=2)

            last_image_dim = tf.shape(image)[-1]
            last_label_dim = tf.shape(gt)[-1]

            combined_crop = tf.image.random_crop(combined, [crop_size[0], crop_size[1], last_image_dim + last_label_dim])

        except tf.errors.InvalidArgumentError:

            # if image height is < crop height after resize then adjust the width of the resized image
            width_res = tf.cast(math.ceil(crop_size[0] * width / height), tf.int32)

            height_res = tf.cast(tf.cast(height, tf.float32)/tf.cast(width, tf.float32)*tf.cast(width_res, tf.float32), tf.int32)
            image_res = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(image, 0), [height_res, width_res]), axis=0)
            gt_res = tf.squeeze(tf.image.resize_nearest_neighbor(tf.expand_dims(gt, 0), [height_res, width_res]), axis=0)

            gt_res = tf.cast(gt_res, tf.float32)

            combined = tf.concat([image_res, gt_res], axis=2)

            last_image_dim = tf.shape(image)[-1]
            last_label_dim = tf.shape(gt)[-1]
            combined_crop = tf.image.random_crop(combined, [crop_size[0], crop_size[1], last_image_dim + last_label_dim])




        image_crop = combined_crop[: , :, :last_image_dim]
        gt_crop = combined_crop[: , :, last_image_dim:]

        image_crop.set_shape((crop_size[0], crop_size[1], 3))
        gt_crop.set_shape((crop_size[0], crop_size[1], 1))

        gt_crop = tf.cast(gt_crop, tf.uint8)


        return image_crop, gt_crop

