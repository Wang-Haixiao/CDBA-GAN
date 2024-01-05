from keras.layers import Activation, Conv2D
import keras.backend as K
import tensorflow as tf
from keras.layers import Layer
class CAM(Layer):
    def __init__(self,
                 gamma_initializer=tf.zeros_initializer(),
                 gamma_regularizer=None,
                 gamma_constraint=None,
                 **kwargs):
        super(CAM, self).__init__(**kwargs)
        self.gamma_initializer = gamma_initializer
        self.gamma_regularizer = gamma_regularizer
        self.gamma_constraint = gamma_constraint

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(1, ),
                                     initializer=self.gamma_initializer,
                                     name='gamma',
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)

        self.built = True

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, input):
        # input_shape = input.get_shape().as_list()
        # _, h, w, filters = input_shape
        ##
        x = input
        vec_a = K.reshape(input, shape=[K.shape(x)[0], K.shape(x)[1] * K.shape(x)[2], K.shape(x)[3]])
        ##
        # vec_a = K.reshape(input, (-1, h * w, filters))
        vec_aT = tf.transpose(vec_a, (0, 2, 1))
        aTa = K.batch_dot(vec_aT, vec_a)
        softmax_aTa = Activation('softmax')(aTa)
        aaTa = K.batch_dot(vec_a, softmax_aTa)
        # aaTa = K.reshape(aaTa, (-1, h, w, filters))
        aaTa = K.reshape(aaTa, (-1,  K.shape(x)[1], K.shape(x)[2], K.shape(x)[3]))
        out = self.gamma*aaTa + input
        return out