from keras.layers import MaxPooling1D, Flatten, Input, BatchNormalization
from keras.layers import Reshape, Dense
from keras.models import Model
from keras.engine.topology import Layer
from utils.keras_wrapped_layers import drop_connect_dense, drop_connect_conv_1d
import numpy as np
import tensorflow as tf


class MatMul(Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list):
            raise ValueError('`MatMul` layer should be called '
                             'on a list of inputs')
        if len(input_shape) != 2:
            raise ValueError('The input of `MatMul` layer should be a list containing 2 elements')

        if len(input_shape[0]) != 3 or len(input_shape[1]) != 3:
            raise ValueError('The dimensions of each element of inputs should be 3')

        if input_shape[0][-1] != input_shape[1][1]:
            raise ValueError('The last dimension of inputs[0] should match the dimension 1 of inputs[1]')

    def call(self, inputs):
        if not isinstance(inputs, list):
            raise ValueError('A `MatMul` layer should be called '
                             'on a list of inputs.')
        return tf.matmul(inputs[0], inputs[1])

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0][0], input_shape[0][1], input_shape[1][-1]]
        return tuple(output_shape)


def conv1d_bn(x, nb_filters, kernel_size, is_training=True):
    x = drop_connect_conv_1d(x,
                             nb_filters=nb_filters,
                             kernel_size=kernel_size,
                             activation='relu',
                             drop_prob=0.5,
                             is_training=is_training)
    x = BatchNormalization()(x)
    return x


def dense_bn(x, units, is_training=True):
    x = drop_connect_dense(x, units=units, activation='relu', drop_prob=0.5, is_training=is_training)
    x = BatchNormalization()(x)
    return x


def PointNet_CLS(nb_classes, is_training=True):
    model_inputs = Input(shape=(2048, 3))

    # input transformation net
    x = conv1d_bn(model_inputs, 64, 1, is_training=is_training)
    x = conv1d_bn(x, 128, 1, is_training=is_training)
    x = conv1d_bn(x, 1024, 1, is_training=is_training)
    x = MaxPooling1D(pool_size=2048)(x)

    x = dense_bn(x, 512, is_training=is_training)
    x = dense_bn(x, 256, is_training=is_training)

    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)

    # forward net
    g = MatMul()([model_inputs, input_T])
    g = drop_connect_conv_1d(g, 64, 1, is_training=is_training)
    g = drop_connect_conv_1d(g, 64, 1, is_training=is_training)

    # feature transform net
    f = drop_connect_conv_1d(g, 64, 1, is_training=is_training)
    f = drop_connect_conv_1d(f, 128, 1, is_training=is_training)
    f = drop_connect_conv_1d(f, 1024, 1, is_training=is_training)
    f = MaxPooling1D(pool_size=2048)(f)
    f = dense_bn(f, 512, is_training=is_training)
    f = dense_bn(f, 256, is_training=is_training)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)

    # forward net
    g = MatMul()([g, feature_T])
    g = drop_connect_conv_1d(g, 64, 1, is_training=is_training)
    g = drop_connect_conv_1d(g, 128, 1, is_training=is_training)
    g = drop_connect_conv_1d(g, 1024, 1, is_training=is_training)

    # global feature
    global_feature = MaxPooling1D(pool_size=2048)(g)

    # point_net_cls
    c = dense_bn(global_feature, 512, is_training=is_training)
    c = dense_bn(c, 256, is_training=is_training)
    c = drop_connect_dense(c, nb_classes,
                           activation='softmax',
                           drop_prob=0.2,
                           is_training=is_training)
    c = Flatten()(c)

    model = Model(inputs=model_inputs, outputs=c)
    return model
