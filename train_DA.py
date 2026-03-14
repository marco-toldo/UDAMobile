import tensorflow as tf
from models.model_DA import CycleGAN_patch
from datetime import datetime
import os
import logging
import numpy as np
from utils import tensor_utils
from utils.utils_compact import ImagePool
import time

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_classes', 19, 'number of classes, default: 16 for SY or 19 for GTA')
tf.flags.DEFINE_integer('batch_size', 1, 'batch size, default: 1')
tf.flags.DEFINE_integer('num_steps', 80000, 'training steps, default: 80000')
tf.flags.DEFINE_integer('crop_height', 800, 'training crop height, default: 600')
tf.flags.DEFINE_integer('crop_width', 800, 'training crop width, default: 600')
tf.flags.DEFINE_integer('width_res', 1536, 'width after resize when keeping original image aspect ratio, default: 1280 for SY or 1536 for GTA')

tf.flags.DEFINE_float('lambda_cycle_G', 10., 'weight for forward cycle loss (X->Y->X), default: 10')
tf.flags.DEFINE_float('lambda_cycle_F', 10., 'weight for backward cycle loss (Y->X->Y), default: 10')
tf.flags.DEFINE_float('lambda_feat', 0.0001, 'weight for adversarial feature losses, default: 1e-4')
tf.flags.DEFINE_float('lambda_cross', 1.0, 'weight for cross-entropy loss, default: 1.0')
tf.flags.DEFINE_float('lambda_sem', 0.1, 'weight for semantic loss, default: 0.1')
tf.flags.DEFINE_float('lambda_iden', 0.0, 'weight for identity loss, default: 0.0')


tf.flags.DEFINE_integer('ngf', 64, 'number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('norm', 'instance', '[instance, batch] use instance norm or batch norm, default: instance')
tf.flags.DEFINE_bool('use_lsgan', True, 'use lsgan or cross entropy loss, default: True')
tf.flags.DEFINE_float('pool_size', 50, 'size of image buffer that stores previously generated images, default: 50')

tf.flags.DEFINE_multi_integer('stride_list', [1,1,1,1], 'strides on the feature discriminators, default: [1,1,1,1]') # strides -> receptive field: 1-1-1-1 -> 16, 2-1-1-1 -> 28, 2-2-1-1 -> 46
tf.flags.DEFINE_multi_integer('channel_list', [64, 128, 256, 512], 'output channels of the feature discriminators, default: [64, 128, 256, 512]')
tf.flags.DEFINE_string('features_to_use', 'before_aspp', '[before_aspp, after_aspp, concatenated, before_aspp_mean], default: before_aspp')


tf.flags.DEFINE_float('learning_rate_cyclegan', 2e-4, 'initial learning rate for cyclegan Adam, default: 2e-4')
tf.flags.DEFINE_float('learning_rate_feature_discriminator', 2e-4, 'initial learning rate for feature discriminators Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1_cyclegan', 0.5, 'beta1 for cyclegan optimizer, default: 0.5')
tf.flags.DEFINE_float('beta1_feature_discriminator', 0.5, 'beta1 for feat discriminators optimizer, default: 0.5')

tf.flags.DEFINE_float('learning_rate_segmentator', 1e-05, 'initial learning rate of segmentator Adam, default: 5e-06 for SY or 1e-05 for GTA')
tf.flags.DEFINE_float('learning_power_segmentator', 0.9, 'decay power of segmentator learning rate, default: 0.9')
tf.flags.DEFINE_integer('start_decay_step_segmentator', 20000, 'start of segmentator learning rate decay: default: 20000')
tf.flags.DEFINE_integer('decay_steps_segmentator', 60000, 'decay steps of segmentator learning rate, default: 60000')
tf.flags.DEFINE_integer('start_feat_step', 20000, 'start of feature alignment, default: 20000')
tf.flags.DEFINE_float('end_learning_rate_segmentator', 0.0, 'final learning rate of segmentator Adam ')
tf.flags.DEFINE_float('beta1_segmentator', 0.9, 'momentum term of segmentator Adam, default: 0.5')
tf.flags.DEFINE_float('weight_decay_segmentator', 0.00004, 'segmentator weight decay, default: 0.00004')

tf.flags.DEFINE_string('domain_val', 'CS', 'dataset used for validation')
tf.flags.DEFINE_bool('save_images', True, 'save images or not, default: True')
tf.flags.DEFINE_bool('debug', False, 'debug or not, default: False')


tf.flags.DEFINE_string('X_tfrecords', '../../Datasets/GTA/tfrecords/train_div_2.tfrecords', 'path to tfrecords of source data')
tf.flags.DEFINE_string('Y_tfrecords', '../../Datasets/GTA/tfrecords/val_2.tfrecords', 'path to tfrecords of target data')
tf.flags.DEFINE_string('pretrained_model_seg', '../../Pretrained/mobilenet/GTA_lr04/model90000.ckpt', 'path to checkpoint of segmentator source pre-trained weights')


tf.flags.DEFINE_integer('val_interval', 2000, 'interval between validations')
tf.flags.DEFINE_integer('val_samples', 500, 'samples used for validation')
tf.flags.DEFINE_multi_integer('val_crop_size',[1024, 2048], 'size of validation images')
tf.flags.DEFINE_string('validation_tfrecords', '../../Datasets/Cityscapes/tfrecords/Cityscapes_mapped_val.tfrecords', 'path to tfrecords of validation data')


tf.flags.DEFINE_string('load_model', None, 'path to saved checkpoint to be loaded, if None standard initialization is performed')



def train_cyclegan(checkpoint_seg, current_time_precomputed):

  if FLAGS.load_model is not None:
    checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
    current_time = current_time_precomputed \
                   + '_DA' \
                   + '_steps' + str(FLAGS.start_feat_step) + '-' + str(FLAGS.num_steps) \
                   + '_size' + str(FLAGS.crop_height) + 'x' + str(FLAGS.crop_width) + '-' + str(FLAGS.width_res) \
                   + '_lr-s' + str(FLAGS.learning_rate_segmentator)

    checkpoints_dir = "checkpoints/{}".format(current_time)

    try:
        os.makedirs(checkpoints_dir + "/segmentator_model/" + FLAGS.domain_val)
    except os.error:
        pass

  graph = tf.Graph()

  with graph.as_default():

    cycle_gan = CycleGAN_patch(
        X_train_file=FLAGS.X_tfrecords,
        Y_train_file=FLAGS.Y_tfrecords,
        validation_file=FLAGS.validation_tfrecords,
        num_classes=FLAGS.num_classes,
        batch_size=FLAGS.batch_size,
        crop_height=FLAGS.crop_height,
        crop_width=FLAGS.crop_width,
        width_res=FLAGS.width_res,
        lambda_cycle_G=FLAGS.lambda_cycle_G,
        lambda_cycle_F=FLAGS.lambda_cycle_F,
        lambda_feat=FLAGS.lambda_feat,
        lambda_cross=FLAGS.lambda_cross,
        lambda_sem=FLAGS.lambda_sem,
        lambda_iden=FLAGS.lambda_iden,
        ngf=FLAGS.ngf,
        norm=FLAGS.norm,
        use_lsgan=FLAGS.use_lsgan,
        stride_list=FLAGS.stride_list,
        channel_list=FLAGS.channel_list,
        features_to_use=FLAGS.features_to_use,
        learning_rate_cyclegan=FLAGS.learning_rate_cyclegan,
        learning_rate_feature_discriminator=FLAGS.learning_rate_feature_discriminator,
        beta1_cyclegan=FLAGS.beta1_cyclegan,
        beta1_feature_discriminator=FLAGS.beta1_feature_discriminator,
        learning_rate_segmentator=FLAGS.learning_rate_segmentator,
        learning_power_segmentator=FLAGS.learning_power_segmentator,
        start_decay_step_segmentator=FLAGS.start_decay_step_segmentator,
        decay_steps_segmentator=FLAGS.decay_steps_segmentator,
        end_learning_rate_segmentator=FLAGS.end_learning_rate_segmentator,
        beta1_segmentator=FLAGS.beta1_segmentator,
        weight_decay_segmentator=FLAGS.weight_decay_segmentator,
        save_images=FLAGS.save_images,
        debug=FLAGS.debug,
    )

    # the ops to compute the losses and the outputs of G and F when a batch of images is fed to the network
    G_loss_without_feat, G_loss_with_feat, D_img_Y_loss, D_feat_Y_loss, F_loss_without_feat, F_loss_with_feat, D_img_X_loss, D_feat_X_loss, S_task_loss, S_loss_with_task, fake_y, fake_x, loss_list, training_summary = cycle_gan.model()
    iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, mean_IoU, mean_IoU_update_op, validation_summary = cycle_gan.model_val()


    # different networks to be optimized:
    # -> 'cyclegan_and_segmentator': all losses to train all networks inside the model
    # -> 'cyclegan_with_feat': S not trainable
    # -> 'cyclegan_without_feat': cyclegan standard + semantic loss
    # -> 'segmentator_only': only task loss for S

    optimizers_without_feat, optimizer_summary_without_feat = cycle_gan.optimize(G_loss_without_feat,
                                                                                   D_img_Y_loss,
                                                                                   D_feat_Y_loss,
                                                                                   F_loss_without_feat,
                                                                                   D_img_X_loss,
                                                                                   D_feat_X_loss,
                                                                                   S_loss_with_task,
                                                                                   network='cyclegan_without_feat',
                                                                                   name='opt_without_S')

    optimizers_with_feat, optimizer_summary_with_feat = cycle_gan.optimize(G_loss_with_feat,
                                                                            D_img_Y_loss,
                                                                            D_feat_Y_loss,
                                                                            F_loss_with_feat,
                                                                            D_img_X_loss,
                                                                            D_feat_X_loss,
                                                                            S_loss_with_task,
                                                                            network='cyclegan_and_segmentator',
                                                                            name='opt_with_S')

    train_summary = tf.summary.merge([optimizer_summary_without_feat, optimizer_summary_with_feat, training_summary])


    train_writer = tf.summary.FileWriter(checkpoints_dir, graph)  # writer to the specific directory for variables in 'graph'
    saver = tf.train.Saver(max_to_keep=2)
    seg_saver = tf.train.Saver(var_list=cycle_gan.S.all_var, max_to_keep=2)

    logging.info('--------------Variables:---------------')
    logging.info('  S         : {}'.format(len(cycle_gan.S.all_var)))
    logging.info('  G         : {}'.format(len(cycle_gan.G.variables)))
    logging.info('  F         : {}'.format(len(cycle_gan.F.variables)))
    logging.info('  D_feat_X  : {}'.format(len(cycle_gan.D_feat_X.variables)))
    logging.info('  D_feat_Y  : {}'.format(len(cycle_gan.D_feat_Y.variables)))
    logging.info('  D_img_X   : {}'.format(len(cycle_gan.D_img_X.variables)))
    logging.info('  D_img_Y   : {}'.format(len(cycle_gan.D_img_Y.variables)))

    if FLAGS.debug:
        for i in cycle_gan.S.all_var:
            if 'biases' in i.name or 'weights' in i.name:
                print(i)
        for i in cycle_gan.G.variables:
            if 'biases' in i.name or 'weights' in i.name:
                print(i)
        for i in cycle_gan.D_img_X.variables:
            if 'biases' in i.name or 'weights' in i.name:
                print(i)

  with tf.Session(graph=graph) as sess:

    if FLAGS.load_model is not None:
      checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
      meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
      print(meta_graph_path)
      print(meta_graph_path.split("/")[2].split("-")[2].split(".")[0])
      step = int(meta_graph_path.split("/")[2].split("-")[2].split(".")[0])  # recover the global step value from the name

    else:
      sess.run(tf.global_variables_initializer())
      seg_loader = tf.train.Saver(var_list=cycle_gan.S.all_var, reshape=True)
      seg_loader.restore(sess, checkpoint_seg)
      step = 0


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    mIoU_max = 0.0

    try:
      fake_Y_pool = ImagePool(FLAGS.pool_size)
      fake_X_pool = ImagePool(FLAGS.pool_size)

      # repeat the optimization process multiple times
      while (not coord.should_stop()) and (step < FLAGS.num_steps):


        seconds_start = time.time()

        ### VALIDATION ###
        if step % FLAGS.val_interval == 0 and (step == 0 or step == FLAGS.val_interval or step > FLAGS.start_feat_step-1):
            sess.run(tf.local_variables_initializer())
            sess.run([iterator.initializer])
            logging.info(' Step : {}'.format(step))
            n = 0
            local_confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int64)
            for i in range(FLAGS.val_samples):
                try:
                    cm, _, _, _ = sess.run(
                       [confusion_matrix, mean_validation_update_op, mean_IoU_update_op, accuracy_update_op])
                    local_confusion_matrix += cm
                    n += 1
                except tf.errors.OutOfRangeError:  # no more items
                    logging.info(
                        ' The chosen value ({}) exceed number of validation images! Used {} instead.'.format(
                          FLAGS.val_samples, n))
                    break
            mean_validation_loss_val, mean_IoU_val, accuracy_val, summary_val = sess.run(
                [mean_validation_loss, mean_IoU, accuracy, validation_summary])
            train_writer.add_summary(summary_val, step)
            train_writer.flush()
            logging.info(' Validation_loss : {}'.format(mean_validation_loss_val))
            logging.info(' accuracy : {:0.2f}%'.format(accuracy_val * 100))
            logging.info(' mIoU : {:0.2f}%'.format(mean_IoU_val * 100))

            tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix, FLAGS.num_classes,
                                                         class_mask=None)

            if mean_IoU_val > mIoU_max and step != 0:
                mIoU_max = mean_IoU_val
                model_mIoU = str('{:0.2f}'.format(mean_IoU_val * 100))
                model_mIoU = "".join(model_mIoU.split("."))
                save_path = seg_saver.save(sess,
                                           checkpoints_dir + "/segmentator_model/" + FLAGS.domain_val1 + "/model" + model_mIoU + ".ckpt",
                                           global_step=step)
                logging.info("Segmentation model saved in file: %s" % save_path)

            seconds_end = time.time()
            print("Seconds since validation start =", seconds_end - seconds_start)


        ### TRAINING ###
        fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

        if step < FLAGS.start_feat_step:
            _, G_loss_val, D_img_Y_loss_val, D_feat_Y_loss_val, F_loss_val, D_img_X_loss_val, D_feat_X_loss_val, S_loss_with_task_val, loss_list_val, summary = (
                sess.run(
                  [optimizers_without_feat, G_loss_without_feat, D_img_Y_loss, D_feat_Y_loss, F_loss_without_feat, D_img_X_loss, D_feat_X_loss, S_loss_with_task, loss_list, train_summary], # running 'optimizers' gives no output (just optimization)
                  feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                             cycle_gan.fake_x: fake_X_pool.query(fake_x_val),
                             cycle_gan.variable_step: step}
                )
            )
        else:
            _, G_loss_val, D_img_Y_loss_val, D_feat_Y_loss_val, F_loss_val, D_img_X_loss_val, D_feat_X_loss_val, S_loss_with_task_val, loss_list_val, summary = (
                sess.run(
                    [optimizers_with_feat, G_loss_with_feat, D_img_Y_loss, D_feat_Y_loss, F_loss_with_feat, D_img_X_loss, D_feat_X_loss, S_loss_with_task, loss_list, train_summary],
                    feed_dict={cycle_gan.fake_y: fake_Y_pool.query(fake_y_val),
                               cycle_gan.fake_x: fake_X_pool.query(fake_x_val),
                               cycle_gan.variable_step: step}
                )
            )

        if step % 100 == 0:
            train_writer.add_summary(summary, step)
            train_writer.flush()  # Forces summary writer to send any buffered data to storage.

        if step == FLAGS.start_feat_step:
            logging.info('--------------PHASE 2-------------------')

        if step % 100 == 0:
          logging.info('--------------Step %d:-----------------' % step)
          logging.info('  G_loss           : {}'.format(G_loss_val))
          logging.info('  D_img_Y_loss     : {}'.format(D_img_Y_loss_val))
          logging.info('  D_feat_Y_loss    : {}'.format(D_feat_Y_loss_val))
          logging.info('  F_loss           : {}'.format(F_loss_val))
          logging.info('  D_img_X_loss     : {}'.format(D_img_X_loss_val))
          logging.info('  D_feat_X_loss    : {}'.format(D_feat_X_loss_val))
          logging.info('-----------Single Losses----------------')
          logging.info('  G_img_adv        : {}'.format(loss_list_val.pop(0)))
          logging.info('  G_feat_adv       : {}'.format(loss_list_val.pop(0)))
          logging.info('  F_img_adv        : {}'.format(loss_list_val.pop(0)))
          logging.info('  F_feat_adv       : {}'.format(loss_list_val.pop(0)))
          logging.info('  Cycle_loss       : {}'.format(loss_list_val.pop(0)))
          logging.info('  G_seg            : {}'.format(loss_list_val.pop(0)))
          logging.info('  F_seg            : {}'.format(loss_list_val.pop(0)))
          logging.info('  S_task           : {}'.format(loss_list_val.pop(0)))
          if FLAGS.lambda_iden != 0.:
              logging.info('  Iden_loss        : {}'.format(loss_list_val.pop(0)))



        if step % 10000 == 0 and step != 0:
          save_path = saver.save(sess, checkpoints_dir + "/model.ckpt", global_step=step)  # save variables to a checkpoint file
          logging.info("Model saved in file: %s" % save_path)

        step += 1

    except KeyboardInterrupt:
        logging.info('Interrupted')
        coord.request_stop()
    except Exception as e:
        coord.request_stop(e)
    finally:
        ### VALIDATION ###
        sess.run(tf.local_variables_initializer())
        sess.run([iterator.initializer])
        logging.info(' Step : {}'.format(step))
        n = 0
        local_confusion_matrix = np.zeros((FLAGS.num_classes, FLAGS.num_classes), dtype=np.int64)
        for i in range(FLAGS.val_samples):
            try:
                cm, _, _, _ = sess.run(
                    [confusion_matrix, mean_validation_update_op, mean_IoU_update_op, accuracy_update_op])
                local_confusion_matrix += cm
                n += 1
            except tf.errors.OutOfRangeError:  # no more items
                logging.info(
                    ' The chosen value ({}) exceed number of validation images! Used {} instead.'.format(
                        FLAGS.val_samples, n))
                break
        mean_validation_loss_val, mean_IoU_val, accuracy_val, summary_val = sess.run(
            [mean_validation_loss, mean_IoU, accuracy, validation_summary])
        train_writer.add_summary(summary_val, step)
        train_writer.flush()
        logging.info(' Validation_loss : {}'.format(mean_validation_loss_val))
        logging.info(' accuracy : {:0.2f}%'.format(accuracy_val * 100))
        logging.info(' mIoU : {:0.2f}%'.format(mean_IoU_val * 100))

        tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix, FLAGS.num_classes,
                                                     class_mask=None)

        model_mIoU = str('{:0.2f}'.format(mean_IoU_val * 100))
        model_mIoU_v1 = "".join(model_mIoU.split("."))



        save_path = saver.save(sess, checkpoints_dir + "/model" + model_mIoU_v1 + ".ckpt", global_step=step)
        logging.info("Model saved in file: %s" % save_path)

        coord.request_stop()
        coord.join(threads)


  return checkpoints_dir



def main(unused_argv):

    current_time = datetime.now().strftime("%Y%m%d-%H%M")
    train_cyclegan(
        checkpoint_seg=FLAGS.pretrained_model_seg,
        current_time_precomputed=current_time
    )

if __name__ == '__main__':  # if current file is executed under a shell instead of imported as a module
    logging.basicConfig(level=logging.INFO)
    tf.app.run()