import tensorflow as tf
from models.model_S import S
from datetime import datetime
import os
import logging
import numpy as np
from utils import tensor_utils
import time



FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('num_classes', 19, 'number of classes, default: 16 for SY or 19 for GTA')
tf.flags.DEFINE_integer('batch_size', 5, 'batch size, default: 5')
tf.flags.DEFINE_integer('num_steps', 90000, 'training steps, default: 90000')
tf.flags.DEFINE_integer('crop_height', 800, 'training crop height, default: 800')
tf.flags.DEFINE_integer('crop_width', 800, 'training crop width, default: 800')
tf.flags.DEFINE_integer('width_res', 1914, 'width after resize when keeping original image aspect ratio, default: 1280 for SY or 1914 for GTA')

tf.flags.DEFINE_string('color_dataset', 'Cityscapes', 'Only Cityscapes and SYNTHIA are accepted')
tf.flags.DEFINE_string('domain_val', 'GTA', 'dataset used for validation')
tf.flags.DEFINE_bool('save_images', False, 'save images or not, default: True')

#### LEARNING PARAMS ####
tf.flags.DEFINE_float('learning_rate', 1e-04, 'initial learning rate of segmentator Adam, default: 5e-05 for SY or 1e-06 for GTA')
tf.flags.DEFINE_float('learning_power', 0.9, 'decay power of segmentator learning rate, default: 0.9')
tf.flags.DEFINE_integer('start_decay_step', 0, 'start of segmentator learning rate decay: default: 0')
tf.flags.DEFINE_float('last_layer_gradient_multiplier', 10.0, 'multiplicative factor for last layer to boost their learning in case pre-trained model doesn\'t contain them')
tf.flags.DEFINE_float('beta1', 0.9, 'beta1 term of segmentator Adam')
tf.flags.DEFINE_bool('upsample_logits', True, 'Upsample logits when computing the cross-entropy loss or not')
tf.flags.DEFINE_bool('debug', False, 'debug or not, default: False')



#### DATA ####

tf.flags.DEFINE_string('train_tfrecords', '../../Datasets/GTA/tfrecords/train_div_2.tfrecords', 'path to training data')
tf.flags.DEFINE_integer('validation_interval', 500, 'interval between validations')
tf.flags.DEFINE_integer('val_samples', 500, 'samples used for validation')
tf.flags.DEFINE_string('validation_tfrecords', '../../Datasets/GTA/tfrecords/val_2.tfrecords', 'path to validation data')

tf.flags.DEFINE_string('pretrained_model_seg', '../../Pretrained/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000', 'path to pre-trained segmentator model')


tf.flags.DEFINE_string('load_model', None, 'path to saved checkpoint to be loaded, if None standard initialization is performed')



def train():

  if FLAGS.load_model is not None:
      checkpoints_dir = "checkpoints/" + FLAGS.load_model.lstrip("checkpoints/")
  else:
      current_time = datetime.now().strftime("%Y%m%d-%H%M") \
                     + '_S' \
                     + '_s' + str(FLAGS.num_steps) \
                     + '_b' + str(FLAGS.batch_size) \
                     + '_c' + str(FLAGS.crop_height) + 'x' + str(FLAGS.crop_width)

      checkpoints_dir = "checkpoints/{}".format(current_time)


  try:
      os.makedirs(checkpoints_dir)
  except os.error:
      pass


  graph = tf.Graph()
  with graph.as_default():

      S_network = S(
          width_res=FLAGS.width_res,
          train_crop_size=[FLAGS.crop_height, FLAGS.crop_width],
          num_classes=FLAGS.num_classes,
          X_train_file=FLAGS.train_tfrecords,
          batch_size=FLAGS.batch_size,
          learning_rate=FLAGS.learning_rate,
          start_decay_step=FLAGS.start_decay_step,
          decay_steps=FLAGS.num_steps-FLAGS.start_decay_step,
          beta1=FLAGS.beta1,
          learning_power=FLAGS.learning_power,
          last_layer_gradient_multiplier=FLAGS.last_layer_gradient_multiplier,
          upsample_logits=FLAGS.upsample_logits,
          validation_file=FLAGS.validation_tfrecords,
          save_images=FLAGS.save_images,
          color_dataset=FLAGS.color_dataset,
          debug=FLAGS.debug
      )

      S_loss, training_summary = S_network.model()
      optimizers = S_network.optimize(S_loss)
      iterator, mean_validation_loss, mean_validation_update_op, confusion_matrix, accuracy, accuracy_update_op, mean_IoU, mean_IoU_update_op, validation_summary = S_network.model_val()
      train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
      saver = tf.train.Saver(max_to_keep=2)



  with tf.Session(graph=graph) as sess:

      if FLAGS.load_model is not None:  # restore the saved model
          checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
          meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
          restore = tf.train.import_meta_graph(meta_graph_path)
          restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
          step = int(meta_graph_path.split("-")[2].split(".")[0])
      else:
          sess.run(tf.global_variables_initializer())
          seg_loader = tf.train.Saver(var_list=S_network.S.all_var_without_last_layers, reshape=True)
          seg_loader.restore(sess, FLAGS.pretrained_model_seg)
          step = 0

      mIoU_max = 0.0
      no_improvement_count = 0


      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      try:

          while (not coord.should_stop()) and (step < FLAGS.num_steps):

              ### VALIDATION ###
              if step % FLAGS.validation_interval == 0: #and (step != 0 or val_at_step_zero):

                  seconds_start = time.time()

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

                  if mean_IoU_val > mIoU_max:
                      no_improvement_count = 0
                      mIoU_max = mean_IoU_val
                      model_mIoU = str('{:0.2f}'.format(mean_IoU_val * 100))
                      model_mIoU = "".join(model_mIoU.split("."))
                      save_path = saver.save(sess, checkpoints_dir + "/" + FLAGS.domain_val + "/model" + model_mIoU + ".ckpt", global_step=step)
                      logging.info("Segmentation model saved in file: %s" % save_path)
                  else:
                      no_improvement_count += 1

                  logging.info(' max_mIoU : {:0.2f}%'.format(mIoU_max * 100))

                  seconds_end = time.time()
                  print("Seconds since validation_1 start =", seconds_end - seconds_start)

              ### TRAINING ###
              _, S_loss_val, summary = (
                  sess.run(
                      [optimizers, S_loss, training_summary]
                  )
              )

              if step % 30 == 0:
                  train_writer.add_summary(summary, step)
                  train_writer.flush()

              if step % 100 == 0:
                  logging.info('--------------Step %d:---------------' % step)
                  logging.info('  S_loss        : {}'.format(S_loss_val))


              step += 1

      except KeyboardInterrupt:
          logging.info('Interrupted')
          coord.request_stop()
      except Exception as e:
          coord.request_stop(e)
      finally:

          ### VALIDATION ###
          logging.info(' Step : {}'.format(step))
          sess.run(tf.local_variables_initializer())
          sess.run([iterator.initializer])
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
          mean_validation_loss_val, mean_IoU_val, accuracy_val, summary = sess.run(
              [mean_validation_loss, mean_IoU, accuracy, validation_summary])


          train_writer.add_summary(summary, step)
          train_writer.flush()
          logging.info(' Validation_loss : {}'.format(mean_validation_loss_val))
          logging.info(' accuracy : {:0.2f}%'.format(accuracy_val * 100))
          logging.info(' mIoU : {:0.2f}%'.format(mean_IoU_val * 100))
          tensor_utils.compute_and_print_IoU_per_class(local_confusion_matrix, FLAGS.num_classes, class_mask=None)

          save_path = saver.save(sess, checkpoints_dir + "/model" + str(FLAGS.num_steps) + ".ckpt")
          logging.info('Final model at step {} saved in file: {}'.format(step, save_path))
          logging.info("######## END ########")
          coord.request_stop()
          coord.join(threads)



def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()