import tensorflow as tf
import cyclegan.ops as ops

class Discriminator:
      def __init__(self, name, is_training, debug, norm='instance', use_sigmoid=False):
            self.name = name
            self.is_training = is_training
            self.norm = norm
            self.reuse = False
            self.use_sigmoid = use_sigmoid
            self.debug = debug

      def __call__(self, input):

            with tf.variable_scope(self.name):


              #1) convolution layers

              #4x4, stride=2, out_ch=64, leaky ReLU slope=0.2
              C64 = ops.Ck(input, 64, reuse=self.reuse, norm=None,
                  is_training=self.is_training, name='C64')

              # 4x4, stride=2, out_ch=128, leaky ReLU slope=0.2
              C128 = ops.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                  is_training=self.is_training, name='C128')

              # 4x4, stride=2, out_ch=256, leaky ReLU slope=0.2
              C256 = ops.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                  is_training=self.is_training, name='C256')

              # 4x4, stride=2, out_ch=512, leaky ReLU slope=0.2
              C512 = ops.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
                  is_training=self.is_training, name='C512')


              #2) last conv layer

              # apply a convolution to produce a  h/16 x w/16  dimensional output (1 output channel)
              # use_sigmoid = False if use_lsgan = True
              #4x4, stride=1, out_ch=1
              output = ops.last_conv(C512, reuse=self.reuse,
                  use_sigmoid=self.use_sigmoid, name='output')

              if self.debug:
                  print('discriminator output shape:')
                  print(output.get_shape())


            self.reuse = True

            self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)


            return output