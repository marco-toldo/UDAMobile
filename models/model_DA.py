import tensorflow as tf
import cyclegan.ops as ops
from utils import utils_compact
import utils.optimizers as opt
from utils.reader_patch import Patch_Reader
from cyclegan.discriminator_img import Discriminator
from cyclegan.discriminator_feat import Discriminator_feat
from cyclegan.generator import Generator
from cyclegan.segmentator import Segmentator
from core import common
import cyclegan.DeepLabV3 as DeepLabV3


REAL_LABEL = 0.9

class CycleGAN_patch:
    def __init__(self,
                 X_train_file='',
                 Y_train_file='',
                 validation_file='',
                 num_classes=19,
                 min_queue_examples=1000,
                 batch_size=1,
                 crop_height=400,
                 crop_width=400,
                 width_res=1024,
                 lambda_cycle_G=10,
                 lambda_cycle_F=10,
                 lambda_feat=0.001,
                 lambda_cross=1.0,
                 lambda_sem=0.1,
                 lambda_iden=0.0,
                 ngf=64,
                 norm='instance',
                 use_lsgan=True,
                 stride_list=[1, 1, 1, 1],
                 channel_list=[64, 128, 256, 512],
                 features_to_use='before_aspp',
                 learning_rate_cyclegan=2e-4,
                 learning_rate_feature_discriminator=2e-04,
                 learning_power_gen=1.0,
                 learning_power_dis=1.0,
                 start_decay_step_cyclegan=300000,
                 decay_steps_cyclegan=300000,
                 end_learning_rate_cyclegan=0.0,
                 beta1_cyclegan=0.5,
                 beta1_feature_discriminator=0.5,
                 learning_rate_segmentator=1e-04,
                 learning_power_segmentator=0.9,
                 start_decay_step_segmentator=10000,
                 decay_steps_segmentator=50000,
                 end_learning_rate_segmentator=0.0,
                 beta1_segmentator=0.9,
                 weight_decay_segmentator=0.00004,
                 save_images=True,
                 debug=True,
                 ):

        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.validation_file = validation_file

        self.num_classes = num_classes
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.width_res = width_res

        self.lambda_cycle_G = lambda_cycle_G
        self.lambda_cycle_F = lambda_cycle_F
        self.lambda_feat = lambda_feat
        self.lambda_cross = lambda_cross
        self.lambda_sem = lambda_sem
        self.lambda_iden = lambda_iden


        self.ngf = ngf
        self.norm = norm
        self.use_lsgan = use_lsgan
        self.use_sigmoid = not use_lsgan
        self.stride_list = stride_list
        self.channel_list = channel_list
        self.features_to_use = features_to_use


        self.learning_rate_cyclegan = learning_rate_cyclegan
        self.learning_rate_feature_discriminator = learning_rate_feature_discriminator
        self.learning_power_gen = learning_power_gen
        self.learning_power_dis = learning_power_dis
        self.start_decay_step_cyclegan = start_decay_step_cyclegan
        self.decay_steps_cyclegan = decay_steps_cyclegan
        self.end_learning_rate_cyclegan = end_learning_rate_cyclegan
        self.beta1_cyclegan = beta1_cyclegan
        self.beta1_feature_discriminator = beta1_feature_discriminator


        self.learning_rate_segmentator = learning_rate_segmentator
        self.learning_power_segmentator = learning_power_segmentator
        self.start_decay_step_segmentator = start_decay_step_segmentator
        self.decay_steps_segmentator = decay_steps_segmentator
        self.end_learning_rate_segmentator = end_learning_rate_segmentator
        self.beta1_segmentator = beta1_segmentator
        self.weight_decay_segmentator = weight_decay_segmentator

        self.save_images = save_images
        self.debug = debug


        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.variable_step = tf.placeholder_with_default(0, shape=[], name='variable_step')

        possible_features_to_use = ['before_aspp',
                                    'after_aspp',
                                    'concatenated',
                                    'before_aspp_mean']
        if self.features_to_use not in possible_features_to_use:
            raise ValueError('accepted feature names: before_aspp, after_aspp, concatenated', 'before_aspp_mean')


        self.G = Generator('G_generator',
                           self.is_training,
                           ngf=self.ngf,
                           norm=self.norm,
                           image_height=self.crop_height,
                           image_width=self.crop_width)
        self.D_img_Y = Discriminator('D_discriminator_img_Y',
                                     self.is_training,
                                     norm=self.norm,
                                     use_sigmoid=self.use_sigmoid,
                                     debug=self.debug)
        self.D_feat_Y = Discriminator_feat('D_discriminator_feat_Y',
                                           self.is_training,
                                           norm=self.norm,
                                           use_sigmoid=self.use_sigmoid,
                                           stride_list=self.stride_list,
                                           channel_list=self.channel_list,
                                           debug=self.debug)

        self.F = Generator('F_generator',
                           self.is_training,
                           ngf=self.ngf,
                           norm=self.norm,
                           image_height=self.crop_height,
                           image_width=self.crop_width)
        self.D_img_X = Discriminator('D_discriminator_img_X',
                                     self.is_training,
                                     norm=self.norm,
                                     use_sigmoid=self.use_sigmoid,
                                     debug=self.debug)
        self.D_feat_X = Discriminator_feat('D_discriminator_feat_X',
                                           self.is_training,
                                           norm=self.norm,
                                           use_sigmoid=self.use_sigmoid,
                                           stride_list=self.stride_list,
                                           channel_list=self.channel_list,
                                           debug=self.debug)


        self.fake_x = tf.placeholder(tf.float32, shape=[self.batch_size, self.crop_height, self.crop_width, 3])
        self.fake_y = tf.placeholder(tf.float32, shape=[self.batch_size, self.crop_height, self.crop_width, 3])


        self.S = Segmentator(name='S',
                             image_height=self.crop_height,
                             image_width=self.crop_width,
                             num_classes=self.num_classes,
                             weight_decay=self.weight_decay_segmentator,
                             trainable=True,
                             debug=self.debug)



    def model(self):
        # X_train_file: tfrecords file path
        X_reader = Patch_Reader(self.X_train_file, name='X', width_res=self.width_res,
                                crop_size=[self.crop_height, self.crop_width],
                                batch_size=self.batch_size, min_queue_examples=self.min_queue_examples)
        # Y_train_file: tfrecords file path
        Y_reader = Patch_Reader(self.Y_train_file, name='Y', width_res=self.width_res,
                                crop_size=[self.crop_height, self.crop_width],
                                batch_size=self.batch_size, min_queue_examples=self.min_queue_examples)

        x, x_labels = X_reader.feed()  # returns a batch of images in float ([-1,1]) format ([batch_size, image_width, image_height, image_depth])
        y, _ = Y_reader.feed()




        fake_y = self.G(x)
        fake_x = self.F(y)
        fake_y_feat = self.S(fake_y)[1][self.features_to_use]
        fake_x_feat = self.S(fake_x)[1][self.features_to_use]
        y_feat = self.S(y)[1][self.features_to_use]
        x_feat = self.S(x)[1][self.features_to_use]


        ## CycleGAN losses ##

        # 1) Cycle-consistency
        cycle_loss = self.cycle_consistency_loss(self.G, self.F, x, y)  # lam1*mean(abs(F(G(x))-x)) + lam2*mean(abs(G(F(y))-y))

        # 2) Semantic
        S_x_loss = self.semantic_consistency_loss(S_input=self.G(x), gt_info_generated=self.S(x)[0])
        S_y_loss = self.semantic_consistency_loss(S_input=self.F(y), gt_info_generated=self.S(y)[0])

        # 3) Adversarial image-level
        G_gan_img_loss = self.generator_loss(self.D_img_Y, fake_y, use_lsgan=self.use_lsgan)
        D_img_Y_loss = self.discriminator_loss(self.D_img_Y, y, self.fake_y, use_lsgan=self.use_lsgan)
        F_gan_img_loss = self.generator_loss(self.D_img_X, fake_x, use_lsgan=self.use_lsgan)
        D_img_X_loss = self.discriminator_loss(self.D_img_X, x, self.fake_x, use_lsgan=self.use_lsgan)

        G_loss_without_feat = G_gan_img_loss + cycle_loss + S_x_loss
        F_loss_without_feat = F_gan_img_loss + cycle_loss + S_y_loss

        # 4) Adversarial feature-level
        G_gan_feat_loss = self.lambda_feat * self.generator_loss(self.D_feat_Y, fake_y_feat, use_lsgan=self.use_lsgan)
        D_feat_Y_loss = self.lambda_feat * self.discriminator_loss(self.D_feat_Y, y_feat, fake_y_feat, use_lsgan=self.use_lsgan)
        F_gan_feat_loss = self.lambda_feat * self.generator_loss(self.D_feat_X, fake_x_feat, use_lsgan=self.use_lsgan)
        D_feat_X_loss = self.lambda_feat * self.discriminator_loss(self.D_feat_X, x_feat, fake_x_feat, use_lsgan=self.use_lsgan)

        # 5) Identity
        if self.lambda_iden != 0.:
            iden_loss = self.identity_loss(self.G, self.F, x, y)
            G_loss_without_feat = G_loss_without_feat + iden_loss
            F_loss_without_feat = F_loss_without_feat + iden_loss



        G_loss_with_feat = G_loss_without_feat + G_gan_feat_loss
        F_loss_with_feat = F_loss_without_feat + F_gan_feat_loss


        ## Segmentator losses ##
        S_task_loss = self.cross_entropy_loss(S_output=self.S(self.G(x))[0], gt_labels=x_labels) + self.S.l2_loss
        S_loss = S_x_loss + S_y_loss + G_gan_feat_loss + F_gan_feat_loss
        S_loss_with_task = S_loss + S_task_loss



        ## Loss List ##
        loss_list = [G_gan_img_loss, G_gan_feat_loss, F_gan_img_loss, F_gan_feat_loss, cycle_loss, S_x_loss, S_y_loss, S_loss_with_task]
        if self.lambda_iden != 0.:
            loss_list.append(iden_loss)


        # summary (save data to be analyzed with tensorboard)

        training_summary_list = [
            tf.summary.histogram('D_img_Y/true', self.D_img_Y(y)),
            tf.summary.histogram('D_img_Y/fake', self.D_img_Y(fake_y)),
            tf.summary.histogram('D_feat_Y/true', self.D_feat_Y(y_feat)),
            tf.summary.histogram('D_feat_Y/fake', self.D_feat_Y(fake_y_feat)),
            tf.summary.histogram('D_img_X/true', self.D_img_X(x)),
            tf.summary.histogram('D_img_X/fake', self.D_img_X(fake_x)),
            tf.summary.histogram('D_feat_X/true', self.D_feat_X(x_feat)),
            tf.summary.histogram('D_feat_X/fake', self.D_feat_X(fake_x_feat)),
            tf.summary.scalar('losses/G_feat', G_gan_feat_loss),
            tf.summary.scalar('losses/G_img', G_gan_img_loss),
            tf.summary.scalar('losses/D_feat_Y', D_feat_Y_loss),
            tf.summary.scalar('losses/D_img_Y', D_img_Y_loss),
            tf.summary.scalar('losses/F_feat', F_gan_feat_loss),
            tf.summary.scalar('losses/F_img', F_gan_img_loss),
            tf.summary.scalar('losses/D_feat_X', D_feat_X_loss),
            tf.summary.scalar('losses/D_img_X', D_img_X_loss),
            tf.summary.scalar('losses/cycle', cycle_loss),
            tf.summary.scalar('losses/S_x', S_x_loss),
            tf.summary.scalar('losses/S_y', S_y_loss),
            tf.summary.scalar('losses/S_without_task', S_loss),
            tf.summary.scalar('losses/S_only_task', S_task_loss),
            tf.summary.scalar('losses/S_reg', self.S.l2_loss)]

        if self.lambda_iden != 0.:
            training_summary_list =  training_summary_list + [tf.summary.scalar('losses/iden', iden_loss)]

        if self.save_images:
            training_summary_list = training_summary_list + \
                                    [tf.summary.image('X/synthetic', tf.expand_dims(utils_compact.convert2int(x)[0], 0)),
                                    tf.summary.image('X/generated', tf.expand_dims(utils_compact.convert2int(self.G(x))[0], 0)),
                                    tf.summary.image('X/reconstruction', tf.expand_dims(utils_compact.convert2int(self.F(self.G(x)))[0], 0)),
                                    tf.summary.image('X/generated_mask', tf.expand_dims(tf.expand_dims(utils_compact.convert2mask(self.G(x)), 3)[0], 0)),
                                    tf.summary.image('Y/real', tf.expand_dims(utils_compact.convert2int(y)[0], 0)),
                                    tf.summary.image('Y/generated', tf.expand_dims(utils_compact.convert2int(self.F(y))[0], 0)),
                                    tf.summary.image('Y/reconstruction', tf.expand_dims(utils_compact.convert2int(self.G(self.F(y)))[0], 0)),
                                    tf.summary.image('Y/generated_mask', tf.expand_dims(tf.expand_dims(utils_compact.convert2mask(self.F(y)), 3)[0], 0))]


        training_summary = tf.summary.merge(training_summary_list)


        return G_loss_without_feat, G_loss_with_feat, D_img_Y_loss, D_feat_Y_loss, F_loss_without_feat, F_loss_with_feat, D_img_X_loss, D_feat_X_loss, S_task_loss, S_loss_with_task, fake_y, fake_x, loss_list, training_summary



    def optimize(self, G_loss, D_img_Y_loss, D_feat_Y_loss, F_loss, D_img_X_loss, D_feat_X_loss, S_loss, network='cyclegan_only', name='opt'):


        with tf.variable_scope(name):

            if network=='segmentator_only':

                S_optimizer, optimizer_summary = opt.Adam_optimizer_poly(S_loss,
                                                      self.S.all_var,
                                                      decay_power=self.learning_power_segmentator,
                                                      starter_learning_rate=self.learning_rate_segmentator,
                                                      end_learning_rate=self.end_learning_rate_segmentator,
                                                      start_decay_step=self.start_decay_step_segmentator,
                                                      decay_steps=self.decay_steps_segmentator,
                                                      beta1=self.beta1_segmentator,
                                                      global_step=self.variable_step,
                                                      name_optimizer='Adam_S')

                with tf.control_dependencies([S_optimizer, optimizer_summary]):
                    return tf.no_op(name='optimizers_only_S'), optimizer_summary

            else:
                # cyclegan original + ->
                G_optimizer, optimizer_summary_v1 = opt.Adam_optimizer_poly(G_loss,
                                                      self.G.variables,
                                                      decay_power=self.learning_power_gen,
                                                      starter_learning_rate=self.learning_rate_cyclegan,
                                                      end_learning_rate=self.end_learning_rate_cyclegan,
                                                      start_decay_step=self.start_decay_step_cyclegan,
                                                      decay_steps=self.decay_steps_cyclegan,
                                                      beta1=self.beta1_cyclegan,
                                                      global_step=self.variable_step,
                                                      name_optimizer='Adam_G')
                D_img_Y_optimizer, optimizer_summary_v2 = opt.Adam_optimizer_poly(D_img_Y_loss,
                                                            self.D_img_Y.variables,
                                                            decay_power=self.learning_power_dis,
                                                            starter_learning_rate=self.learning_rate_cyclegan,
                                                            end_learning_rate=self.end_learning_rate_cyclegan,
                                                            start_decay_step=self.start_decay_step_cyclegan,
                                                            decay_steps=self.decay_steps_cyclegan,
                                                            beta1=self.beta1_cyclegan,
                                                            global_step=self.variable_step,
                                                            name_optimizer='Adam_D_img_Y')
                F_optimizer, optimizer_summary_v3 = opt.Adam_optimizer_poly(F_loss,
                                                                            self.F.variables,
                                                                            decay_power=self.learning_power_gen,
                                                                            starter_learning_rate=self.learning_rate_cyclegan,
                                                                            end_learning_rate=self.end_learning_rate_cyclegan,
                                                                            start_decay_step=self.start_decay_step_cyclegan,
                                                                            decay_steps=self.decay_steps_cyclegan,
                                                                            beta1=self.beta1_cyclegan,
                                                                            global_step=self.variable_step,
                                                                            name_optimizer='Adam_F')
                D_img_X_optimizer, optimizer_summary_v4 = opt.Adam_optimizer_poly(D_img_X_loss,
                                                            self.D_img_X.variables,
                                                            decay_power=self.learning_power_dis,
                                                            starter_learning_rate=self.learning_rate_cyclegan,
                                                            end_learning_rate=self.end_learning_rate_cyclegan,
                                                            start_decay_step=self.start_decay_step_cyclegan,
                                                            decay_steps=self.decay_steps_cyclegan,
                                                            beta1=self.beta1_cyclegan,
                                                            global_step=self.variable_step,
                                                            name_optimizer='Adam_D_img_X')

                if network=='cyclegan_and_segmentator':
                    # -> + feat_discriminators and segmentator
                    S_optimizer, optimizer_summary_v5 = opt.Adam_optimizer_poly(S_loss,
                                                          self.S.all_var,
                                                          decay_power=self.learning_power_segmentator,
                                                          starter_learning_rate=self.learning_rate_segmentator,
                                                          end_learning_rate=self.end_learning_rate_segmentator,
                                                          start_decay_step=self.start_decay_step_segmentator,
                                                          decay_steps=self.decay_steps_segmentator,
                                                          beta1=self.beta1_segmentator,
                                                          global_step=self.variable_step,
                                                          name_optimizer='Adam_S')
                    D_feat_Y_optimizer, optimizer_summary_v6 = opt.Adam_optimizer_poly(D_feat_Y_loss,
                                                                 self.D_feat_Y.variables,
                                                                 decay_power=self.learning_power_dis,
                                                                 starter_learning_rate=self.learning_rate_feature_discriminator,
                                                                 end_learning_rate=self.end_learning_rate_cyclegan,
                                                                 start_decay_step=self.start_decay_step_cyclegan,
                                                                 decay_steps=self.decay_steps_cyclegan,
                                                                 beta1=self.beta1_feature_discriminator,
                                                                 global_step=self.variable_step,
                                                                 name_optimizer='Adam_D_feat_Y')
                    D_feat_X_optimizer, optimizer_summary_v7 = opt.Adam_optimizer_poly(D_feat_X_loss,
                                                                 self.D_feat_X.variables,
                                                                 decay_power=self.learning_power_dis,
                                                                 starter_learning_rate=self.learning_rate_feature_discriminator,
                                                                 end_learning_rate=self.end_learning_rate_cyclegan,
                                                                 start_decay_step=self.start_decay_step_cyclegan,
                                                                 decay_steps=self.decay_steps_cyclegan,
                                                                 beta1=self.beta1_feature_discriminator,
                                                                 global_step=self.variable_step,
                                                                 name_optimizer='Adam_D_feat_X')

                    optimizer_summary = tf.summary.merge([optimizer_summary_v1,
                                                          optimizer_summary_v2,
                                                          optimizer_summary_v3,
                                                          optimizer_summary_v4,
                                                          optimizer_summary_v5,
                                                          optimizer_summary_v6,
                                                          optimizer_summary_v7])

                    with tf.control_dependencies([G_optimizer, D_feat_Y_optimizer, D_img_Y_optimizer, F_optimizer, D_feat_X_optimizer, D_img_X_optimizer, S_optimizer, optimizer_summary]):
                        return tf.no_op(name='optimizers_with_S'), optimizer_summary

                elif network=='cyclegan_with_feat':

                    D_feat_Y_optimizer, optimizer_summary_v5 = opt.Adam_optimizer_poly(D_feat_Y_loss,
                                                                 self.D_feat_Y.variables,
                                                                 decay_power=self.learning_power_dis,
                                                                 starter_learning_rate=self.learning_rate_feature_discriminator,
                                                                 end_learning_rate=self.end_learning_rate_cyclegan,
                                                                 start_decay_step=self.start_decay_step_cyclegan,
                                                                 decay_steps=self.decay_steps_cyclegan,
                                                                 beta1=self.beta1_feature_discriminator,
                                                                 global_step=self.variable_step,
                                                                 name_optimizer='Adam_D_feat_Y')
                    D_feat_X_optimizer, optimizer_summary_v6 = opt.Adam_optimizer_poly(D_feat_X_loss,
                                                                 self.D_feat_X.variables,
                                                                 decay_power=self.learning_power_dis,
                                                                 starter_learning_rate=self.learning_rate_feature_discriminator,
                                                                 end_learning_rate=self.end_learning_rate_cyclegan,
                                                                 start_decay_step=self.start_decay_step_cyclegan,
                                                                 decay_steps=self.decay_steps_cyclegan,
                                                                 beta1=self.beta1_feature_discriminator,
                                                                 global_step=self.variable_step,
                                                                 name_optimizer='Adam_D_feat_X')

                    optimizer_summary = tf.summary.merge([optimizer_summary_v1,
                                                          optimizer_summary_v2,
                                                          optimizer_summary_v3,
                                                          optimizer_summary_v4,
                                                          optimizer_summary_v5,
                                                          optimizer_summary_v6])

                    with tf.control_dependencies([G_optimizer, D_feat_Y_optimizer, D_img_Y_optimizer, F_optimizer, D_feat_X_optimizer, D_img_X_optimizer, optimizer_summary]):
                        return tf.no_op(name='optimizers_without_S'), optimizer_summary

                elif network=='cyclegan_without_feat':

                    optimizer_summary = tf.summary.merge([optimizer_summary_v1,
                                                          optimizer_summary_v2,
                                                          optimizer_summary_v3,
                                                          optimizer_summary_v4])

                    with tf.control_dependencies([G_optimizer, D_img_Y_optimizer, F_optimizer, D_img_X_optimizer, optimizer_summary]):
                        return tf.no_op(name='optimizers_only_cyclegan'), optimizer_summary

                else:
                    raise ValueError('accepted network names: cyclegan_and_segmentator, cyclegan_with_feat, cyclegan_without_feat, segmentator_only')





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

        val_images = tf.cast(val_images, tf.float32)
        val_images = (2.0 / 255.0) * val_images - 1.0

        val_height = next_val_data[2]
        val_width = next_val_data[3]

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

        validation_summary_list = [tf.summary.scalar('validation/validation_loss', mean_validation_loss),
                                   tf.summary.scalar('validation/mean_IoU', mean_IoU),
                                   tf.summary.scalar('validation/pixel_accuracy', accuracy)]
        validation_summary = tf.summary.merge(validation_summary_list)

        return iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, mean_IoU, mean_IoU_update_op, validation_summary


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


    def discriminator_loss(self, D, y, fake_y, use_lsgan=True):

        if use_lsgan:  # (least-squares GAN)
            # use mean squared error
            error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))  # mean( ||D(y) - REAL_LABEL||^2 )
            error_fake = tf.reduce_mean(tf.square(D(fake_y)))  # mean( ||D(fake_y)||^2 )
        else:
            # use cross entropy
            error_real = -tf.reduce_mean(ops.safe_log(D(y)))  # -mean(log(D(y) + eps))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_y)))  # -mean(log(1 - D(fake_y) + eps))

        loss = (error_real + error_fake) / 2  # <-----  (sort of mean value)

        return loss


    def generator_loss(self, D, fake_y, use_lsgan=True):

        if use_lsgan:
            # use mean squared error
            loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))  # mean( ||D(fake_y) - REAL_LABEL||^2 )
        else:
            # heuristic, non-saturating loss
            loss = -tf.reduce_mean(ops.safe_log(D(fake_y))) / 2  # -mean(log(D(fake_y) + eps)) / 2

        return loss


    def cycle_consistency_loss(self, G, F, x, y):

        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))  # tf.abs(x) tensor of same size of x
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))  # 'reduce_mean' is scalar (float)
        loss = self.lambda_cycle_G * forward_loss + self.lambda_cycle_F * backward_loss

        return loss


    def semantic_consistency_loss(self, S_input, gt_info_generated):

        labels = tf.argmax(gt_info_generated, axis=3)
        labels = tf.cast(labels, tf.uint8)
        labels_one_hot = tf.one_hot(labels, self.num_classes)

        S_output = self.S(S_input)[0]

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=S_output, labels=labels_one_hot)
        loss = tf.reduce_mean(loss)

        return self.lambda_sem * loss


    def semantic_consistency_loss_v2(self, S_input, gt_info_generated):

        S_output = self.S(S_input)[0]
        gt_info_generated = tf.nn.softmax(gt_info_generated, axis=-1)
        gt_info_generated = tf.stop_gradient(gt_info_generated)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=S_output, labels=gt_info_generated)
        loss = tf.reduce_mean(loss)

        return self.lambda_sem * loss


    def cross_entropy_loss(self, S_output, gt_labels):

        labels = tf.cast(gt_labels, tf.uint8)
        labels_one_hot = tf.one_hot(labels, self.num_classes)
        S_output = tf.image.resize_bilinear(S_output, [self.crop_height, self.crop_width])

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=S_output, labels=labels_one_hot)
        target_mask = tf.squeeze(tf.less_equal(gt_labels, self.num_classes - 1), axis=3)
        loss_masked = tf.boolean_mask(loss, target_mask)
        loss = tf.reduce_mean(loss_masked)

        return self.lambda_cross * loss


    def identity_loss(self, G, F, x, y):

        identity_G_loss = tf.reduce_mean(tf.abs(G(y)-y)) # tf.abs(x) tensor of same size of x
        identity_F_loss = tf.reduce_mean(tf.abs(F(x)-x)) # 'reduce_mean' is scalar (float)
        loss = identity_G_loss + identity_F_loss

        return self.lambda_iden*loss







