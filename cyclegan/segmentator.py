import tensorflow as tf
from core import common
import cyclegan.DeepLabV3 as DeepLabV3
from utils import utils_compact

slim = tf.contrib.slim


class Segmentator:

    def __init__(self, name, num_classes, image_height, image_width, debug, weight_decay=0.00004, trainable=True):
        self.name = name
        self.reuse = tf.AUTO_REUSE
        self.num_classes = num_classes
        self.image_height = image_height
        self.image_width = image_width
        self.weight_decay = weight_decay
        self.trainable = trainable
        self.debug = debug

    def __call__(self, net_input):

        with tf.name_scope(self.name):
            tf.set_random_seed(12345)
            net, features = self._build_deeplab(net_input)

            # Trainable Variables
            all_trainable = tf.global_variables()


        self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

        ############################
        all_var_filtered = []
        for var in all_trainable:
            if "D_discriminator" not in var.name and "G_generator" not in var.name and "F_generator" not in var.name and "Adam" not in var.name and "power" not in var.name and "global_step" not in var.name and "feature_combiner" not in var.name and "channel_reducer" not in var.name:
                all_var_filtered.append(var)
        self.all_var = all_var_filtered  ###

        all_var_without_batch_norm = []
        for var in self.all_var:
            if "moving" not in var.name:
                all_var_without_batch_norm.append(var)
        self.all_var_without_batch_norm = all_var_without_batch_norm  ###

        all_var_last_layers = []
        all_var_without_last_layers = []
        last_layers = DeepLabV3.get_extra_layer_scopes(last_layers_contain_logits_only=False)
        for var in self.all_var:
            for layer in last_layers:
                if layer in var.op.name:
                    all_var_last_layers.append(var)
                    break
        for var in self.all_var:
            if var not in all_var_last_layers:
                all_var_without_last_layers.append(var)
        self.all_var_without_last_layers = all_var_without_last_layers  ###
        ############################


        all_var_with_features = []
        for var in all_trainable:
            if "D_discriminator" not in var.name and "G_generator" not in var.name and "F_generator" not in var.name and "Adam" not in var.name and "power" not in var.name and "global_step" not in var.name:
                all_var_with_features.append(var)
        self.all_var_with_features = all_var_with_features  ###

        all_var_with_features_without_batch_norm = []
        for var in self.all_var_with_features:
            if "moving" not in var.name:
                all_var_with_features_without_batch_norm.append(var)
        self.all_var_with_features_without_batch_norm = all_var_with_features_without_batch_norm  ###


        ### DEBUG ###
        if self.debug:
            print('GLOBAL VARS')
            for var in all_trainable:
                print(var.name)
            print('LAST LAYERS VARS')
            for var in all_var_last_layers:
                print(var.name)
            print('REMOVED VARS')
            for var in all_trainable:
                if var not in all_var_filtered:
                    print(var.name)


        return net, features


    def _build_deeplab(self, inputs):

        model_options = common.ModelOptions(
            outputs_to_num_classes={common.OUTPUT_TYPE: self.num_classes},
            crop_size=[self.image_height, self.image_width],
            output_stride=16,
            atrous_rates=[6, 12, 18])
        updated_options = model_options._replace(
            add_image_level_feature=True,
            aspp_with_batch_norm=True,
            logits_kernel_size=1,
            decoder_output_stride=[4],
            prediction_with_upsampled_logits=True,
            model_variant='mobilenet_v2')  # Employ MobileNetv2 for fast test.

        outputs_to_scales_to_logits = DeepLabV3.multi_scale_logits(
            images=inputs,
            model_options=updated_options,
            image_pyramid=None,
            weight_decay=self.weight_decay,
            is_training=self.trainable,                  #####################
            fine_tune_batch_norm=False
        )

        # outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
        # semantic prediction) to a dictionary of multi-scale logits names to
        # logits. For each output_type, the dictionary has keys which
        # correspond to the scales and values which correspond to the logits.
        # For example, if `scales` equals [1.0, 1.5], then the keys would
        # include 'merged_logits', 'logits_1.00' and 'logits_1.50'
        output_type_dict = outputs_to_scales_to_logits[common.OUTPUT_TYPE]
        logits = output_type_dict[DeepLabV3.MERGED_LOGITS_SCOPE]


        concat_logits, end_points = DeepLabV3.extract_features(
            images=inputs,
            model_options=updated_options,
            weight_decay=self.weight_decay,
            reuse=tf.AUTO_REUSE,
            is_training=self.trainable,
            fine_tune_batch_norm=False,
            nas_training_hyper_parameters=None
        )

        features_stride_4 = end_points['layer_4']  # last feature map with stride 4 (ch_out = 24)
        features_stride_8 = end_points['layer_7']  # last feature map with stride 8 (ch_out = 32)
        features_stride_16 = end_points['layer_14']  # not last feature map with stride 16 since
                                                     # output_stride=32 and the last stride=2 is removed (ch_out = 96)

        features_before_aspp = end_points['layer_18']
        features_before_aspp_mean = tf.reduce_mean(features_before_aspp, axis=3, keepdims=True)
        features_after_aspp = concat_logits

        features_concatenated = tf.concat([features_before_aspp, features_after_aspp], axis=3)

        features_dict = {'before_aspp': features_before_aspp,
                         'after_aspp': features_after_aspp,
                         'concatenated': features_concatenated,
                         'before_aspp_mean': features_before_aspp_mean}


        if self.debug:
            print('SEGMENTATOR OUTPUT SHAPE')
            print('Shape intermediate features:')
            print(features_stride_4.get_shape())
            print(features_stride_8.get_shape())
            print(features_stride_16.get_shape())
            print('Shape features before aspp:')
            print(features_before_aspp.get_shape())
            print('Shape features after aspp:')
            print(features_after_aspp.get_shape())
            print('Shape features concatenated:')
            print(features_concatenated.get_shape())
            print('Shape features combined:')
            print('Shape features before aspp mean:')
            print(features_before_aspp_mean.get_shape())


        return logits, features_dict




    def sample(self, net_input, height, width, color_dataset):

        image = self.__call__(net_input)[0]
        image = tf.image.resize_bilinear(image, [height, width])
        image = utils_compact.convert_output2rgb(image, dataset=color_dataset)
        image = tf.image.encode_png(tf.squeeze(image, [0]))
        return image


