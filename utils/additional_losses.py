
import tensorflow as tf
import cyclegan.ops as ops
import math



def semantic_consistency_loss_KL(output_original, output_transformed):

    output_original = tf.nn.softmax(output_original)
    output_transformed = tf.nn.softmax(output_transformed)

    p = tf.maximum(output_original, 1e-12)
    q = tf.maximum(output_transformed, 1e-12)

    # KL divergence: D_kl(P||Q) = sum( P(i)*log2(P(i)/Q(i)) ) -> measures the information lost when Q is used in place of P
    loss = tf.reduce_mean(tf.multiply((-1)*p, ops.safe_log(q)/tf.log(2.))) + tf.reduce_mean(tf.multiply((-1)*q, ops.safe_log(p)/tf.log(2.)))  #log_2(n)=log_e(n)/log_e(2)

    return loss



def geometry_loss(gen, input_image):

    #input_image.set_shape([self.batch_size, self.crop_height, self.crop_width, 3])

    radian_angle_90 = 90. * math.pi / 180.
    radian_angle_270 = 270. * math.pi / 180.

    rotated_image = tf.contrib.image.rotate(input_image, angles=radian_angle_90)
    generated_image = gen(input_image)

    rotated_then_generated_then_rotated_back_image = tf.contrib.image.rotate(gen(rotated_image), angles=radian_angle_270)
    rotated_then_generated_image = gen(rotated_image)
    generated_then_rotated_image = tf.contrib.image.rotate(generated_image, angles=radian_angle_90)

    loss = tf.reduce_mean(tf.abs(generated_image - rotated_then_generated_then_rotated_back_image)) + tf.reduce_mean(tf.abs(rotated_then_generated_image - generated_then_rotated_image))

    return loss