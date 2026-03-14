import tensorflow as tf
from utils.reader_patch import Patch_Reader
from cyclegan.segmentator import Segmentator
from utils import utils_compact
from core.deeplab import preprocess_utils
from core import common
from cyclegan import DeepLabV3


class S:

    def __init__(self,
                 width_res=1914,
                 train_crop_size = [800, 800],
                 num_classes=19,
                 X_train_file = '',
                 batch_size = 5,
                 min_queue_examples = 1000,
                 learning_rate = 1e-04,
                 start_decay_step = 0,
                 decay_steps = 90000,
                 beta1 = 0.9,
                 learning_power = 0.9,
                 last_layer_gradient_multiplier = 1.0,
                 upsample_logits = True,
                 validation_file = '',
                 save_images=True,
                 color_dataset='Cityscapes',
                 debug=False
                 ):

        self.width_res = width_res
        self.train_crop_size = train_crop_size
        self.num_classes = num_classes
        self.X_train_file = X_train_file
        self.batch_size = batch_size
        self.min_queue_examples = min_queue_examples
        self.learning_rate = learning_rate
        self.start_decay_step = start_decay_step
        self.decay_steps = decay_steps
        self.beta1 = beta1
        self.learning_power = learning_power
        self.upsample_logits = upsample_logits
        self.validation_file = validation_file
        self.last_layer_gradient_multiplier = last_layer_gradient_multiplier
        self.save_images = save_images

        self.color_dataset = color_dataset
        self.debug = debug


        self.S = Segmentator(name='S', image_height=self.train_crop_size[0], image_width=self.train_crop_size[1],
                             num_classes=self.num_classes, trainable=True, debug=self.debug)


    def model(self):

        X_reader = Patch_Reader(self.X_train_file, name='X', width_res=self.width_res, crop_size=self.train_crop_size,
                                batch_size=self.batch_size, min_queue_examples=self.min_queue_examples)


        x, x_labels = X_reader.feed()  # returns a batch of images in float ([-1,1]) format ([batch_size, image_width, image_height, image_depth])

        S_loss = self.segmentator_loss_supervised(S_output=self.S(x)[0], gt_labels=x_labels)


        if self.save_images:
            training_summary_list =[tf.summary.scalar('loss/S_x',
                                                      S_loss),
                                    tf.summary.image('X/synthetic',
                                                     tf.expand_dims(utils_compact.convert2int(x)[0], 0)),
                                    tf.summary.image('X/segmented_syn',
                                                     tf.expand_dims(utils_compact.convert_output2rgb( tf.image.resize_bilinear(self.S(x)[0], self.train_crop_size), dataset=self.color_dataset )[0], 0)),
                                    tf.summary.image('X/label_syn',
                                                     tf.expand_dims(utils_compact.convert_gt2rgb( tf.image.resize_nearest_neighbor(x_labels, self.train_crop_size), dataset=self.color_dataset )[0], 0)),
                                    tf.summary.image('X/label_syn_orig',
                                                     tf.expand_dims(x_labels[0], 0)) ]
        else:
            training_summary_list = [tf.summary.scalar('loss/S_x', S_loss)]


        training_summary = tf.summary.merge(training_summary_list)


        return S_loss, training_summary




    def model_val(self):

        def _extract_fn(tfrecord):

            # Extract features using the keys set during creation
            features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'input': tf.FixedLenFeature([], tf.string),
                'gt': tf.FixedLenFeature([], tf.string)
            }

            # Obtain back image and labels
            sample = tf.parse_single_example(tfrecord, features)
            image = tf.image.decode_png(sample['input'], channels=3)
            gt = tf.image.decode_png(sample['gt'], channels=1)
            height = sample['height']
            width = sample['width']

            return [image, gt, height, width]

        # Build iterator for the validation dataset
        validation_dataset = tf.data.TFRecordDataset(self.validation_file).map(_extract_fn)
        iterator = validation_dataset.make_initializable_iterator()
        next_val_data = iterator.get_next()


        # Obtain validation data
        val_images = tf.expand_dims(next_val_data[0], axis=0)
        val_labels = tf.expand_dims(next_val_data[1], axis=0)

        val_height = next_val_data[2]
        val_width = next_val_data[3]

        val_images = tf.cast(val_images, tf.float32)
        val_images = (2.0 / 255.0) * val_images - 1.0

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: self.num_classes},
            crop_size=[val_height, val_width],
            output_stride=16,
            atrous_rates=[6, 12, 18])
        updated_options = model_options._replace(
            add_image_level_feature=True,
            aspp_with_batch_norm=True,
            logits_kernel_size=1,
            decoder_output_stride=[4],
            prediction_with_upsampled_logits=True,
            model_variant='mobilenet_v2')
        val_logits = DeepLabV3.predict_labels(val_images, updated_options)[common.OUTPUT_TYPE]

        val_logits = tf.stop_gradient(val_logits)

        val_labels = tf.reshape(val_labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(val_labels, 255))
        one_hot_val_labels = tf.one_hot(val_labels, self.num_classes, on_value=1.0, off_value=0.0)

        loss_masked = tf.losses.softmax_cross_entropy(  # <--------------------
            one_hot_val_labels,
            tf.reshape(val_logits, shape=[-1, self.num_classes]),  # prediction shape: [batch, H, W, num_classes]
            weights=not_ignore_mask,
            scope='validation_loss')

        validation_loss = tf.reduce_mean(loss_masked)
        validation_loss = tf.stop_gradient(validation_loss)

        mean_validation_loss, mean_validation_update_op = tf.metrics.mean(validation_loss)

        mean_IoU, mean_IoU_update_op, accuracy, accuracy_update_op, confusion_matrix = self.compute_evaluation_metrics(
            prediction=val_logits, gt_labels=val_labels, num_of_classes=self.num_classes)

        validation_summary_list = [tf.summary.scalar('validation/validation_loss_same', mean_validation_loss),
                                   tf.summary.scalar('validation/mean_IoU_same', mean_IoU),
                                   tf.summary.scalar('validation/pixel_accuracy_same', accuracy)]
        validation_summary = tf.summary.merge(validation_summary_list)

        return iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, mean_IoU, mean_IoU_update_op, validation_summary


    def optimize(self, S_loss):
        def make_optimizer(loss, variables, pw=0.9, name='Adam'):
            global_step = tf.train.get_or_create_global_step()
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = self.start_decay_step
            decay_steps = self.decay_steps
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                              decay_steps, end_learning_rate,
                                              power=pw),
                    starter_learning_rate
                )

            )
            lr_summ = tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            #################################
            optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
            grads_and_vars = optimizer.compute_gradients(loss, var_list=variables)  # returns a list of (grad, var) couples
            last_layers = DeepLabV3.get_extra_layer_scopes(last_layers_contain_logits_only=False)
            grad_mult = self.get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier = self.last_layer_gradient_multiplier)  # returns a map var to multiplier
            grads_and_vars = tf.contrib.training.multiply_gradients(grads_and_vars, grad_mult)
            learning_step = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            #################################

            return learning_step, lr_summ

        S_optimizer, lr_summary = make_optimizer(S_loss, self.S.all_var_without_batch_norm, pw=self.learning_power, name='Adam_S')

        with tf.control_dependencies([S_optimizer]):
            return tf.no_op(name='optimizers'), lr_summary




    def segmentator_loss_supervised(self, S_output, gt_labels):
        """
        Compute the loss of the generator in a supervised manner

        :param S_output: S(G(x)). 4D tensor: [batch_size, image_width, image_height, 3]
        :param gt_labels: ground-truth. 4D tensor: [batch_size, image_width, image_height, num_class]
        :return: supervised loss of the generator
        """

        logits = S_output
        labels = gt_labels

        if self.upsample_logits:  ####
            # Label is not downsampled, and instead we upsample logits.
            logits = tf.image.resize_bilinear(
                logits,
                preprocess_utils.resolve_shape(labels, 4)[1:3],
                align_corners=True)
            scaled_labels = labels
        else:
            # Label is downsampled to the same size as logits.
            scaled_labels = tf.image.resize_nearest_neighbor(
                labels,
                preprocess_utils.resolve_shape(logits, 4)[1:3],
                align_corners=True)

        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels, 255))
        one_hot_labels = tf.one_hot(scaled_labels, self.num_classes, on_value=1.0, off_value=0.0)

        loss = tf.losses.softmax_cross_entropy(
            one_hot_labels,
            tf.reshape(logits, shape=[-1, self.num_classes]),
            weights=not_ignore_mask
        )

        return loss



    def compute_evaluation_metrics(self, prediction, gt_labels, num_of_classes):

        pred = tf.argmax(prediction, axis=3)
        pred = tf.reshape(pred, [-1])
        gt = tf.reshape(gt_labels, [-1])
        # Ignoring all labels greater than or equal to n_classes.
        mask = tf.less_equal(gt, num_of_classes - 1)
        pred = tf.boolean_mask(pred, mask)
        gt = tf.boolean_mask(gt, mask)
        # mIoU
        mIoU, mIou_update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=num_of_classes)
        # confusion matrix
        confusion_matrix_eval = tf.confusion_matrix(predictions=pred, labels=gt, num_classes=num_of_classes)
        # Pixel accuracy
        accuracy_eval, accuracy_update_op_eval = tf.metrics.accuracy(predictions=pred, labels=gt)

        return mIoU, mIou_update_op, accuracy_eval, accuracy_update_op_eval, confusion_matrix_eval



    def get_model_gradient_multipliers(self, last_layers, last_layer_gradient_multiplier):

        gradient_multipliers = {}

        for var in tf.global_variables():
            if 'biases' in var.op.name:
                gradient_multipliers[var.op.name] = 2.

            for layer in last_layers:
                if layer in var.op.name and 'biases' in var.op.name:
                    gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
                    break
                elif layer in var.op.name:
                    gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
                    break

        return gradient_multipliers