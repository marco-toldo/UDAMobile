import tensorflow as tf
import cyclegan.ops as ops
from utils import utils_compact

class Generator:

    def __init__(self, name, is_training, ngf=64, norm='instance', image_height=252, image_width=504, use_tanh=False):
        self.name = name
        self.reuse = False
        self.ngf = ngf  # output channels of first conv layer
        self.norm = norm
        self.is_training = is_training
        self.image_height = image_height
        self.image_width = image_width
        self.use_tanh = use_tanh


    def __call__(self, input):

        with tf.variable_scope(self.name):


            #1) Conv layers

            # 7x7, num_filters = self.ngf, stride 1,  half padding => input_dims = output_dims
            c7s1_32 = ops.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm, activation='relu',
                                 reuse=self.reuse, name='c7s1_32')

            # 3x3, num_filters = self.ngf*2, stride 2, half padding => output_dims = input_dims/2
            d64 = ops.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='d64')

            # 3x3, num_filters = self.ngf*4, stride 2, half padding => output_dims = input_dims/2
            d128 = ops.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
                          reuse=self.reuse, name='d128')


            #2) Residual blocks

            # Residual block: input ----------------------------------------------------------------------------->   output =
            #                 input --> conv1 (3x3, stride1, out_ch=in_ch, half pad) --> conv2 --> normalized2 -->   input+normalized2
            if self.image_height <= 128:
                # use 6 residual blocks for 128x128 images
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=6)
            else:
                # 9 blocks for higher resolution
                res_output = ops.n_res_blocks(d128, reuse=self.reuse, n=9)


            #3) Fractional-strided convolution

            # 3x3, stride 1/2, out_channels = 2*self.ngf => output_dims = 2*input_dims
            u64 = ops.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u64')

            # 3x3, stride 1/2, out_channels = 2*self.ngf => output_dims = 2*input_dims
            u32 = ops.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                         reuse=self.reuse, name='u32', output_h=self.image_height, output_w=self.image_width)


            #4) Conv layer

            # 7x7, num_filters = 3, stride 1,  half padding => input_dims = output_dims
            if self.use_tanh:
                output = ops.c7s1_k(u32, 3, norm=None,
                                    activation='tanh', reuse=self.reuse, name='output')
            else:
                output = ops.c7s1_k(u32, 3, norm=None,
                                    activation='none', reuse=self.reuse, name='output')


        self.reuse = True

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output


    def sample(self, input):

        # float: pixel in [-1,1], int: pixel in [0,255]

        image = utils_compact.convert2int(self.__call__(input)[0])
        image = tf.image.encode_png(image)


        return image