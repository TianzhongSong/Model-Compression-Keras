"""
Drop connect for conv1d, conv2d, conv3d, fc layer
author: Tianzhong
08/27/2018
"""
from keras.layers import Conv1D, Conv2D, Conv3D, Dense
from utils.keras_drop_connect_layer import DropConnect


def drop_connect_conv_1d(x,
                         nb_filters,
                         kernel_size,
                         strides=1,
                         padding='valid',
                         dilation_rate=1,
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         drop_prob=0.,
                         is_training=True):
    """
    Drop connect for 1D convolution layer

    drop_prob: drop rate for weight kernel
    is_training: bool, True for training, False for test
    """
    x = DropConnect(Conv1D(nb_filters,
                           kernel_size,
                           strides=strides,
                           padding=padding,
                           dilation_rate=dilation_rate,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint),
                    prob=drop_prob, training=is_training)(x)

    return x


def drop_connect_conv_2d(x,
                         nb_filters,
                         kernel_size,
                         strides=(1, 1),
                         padding='valid',
                         dilation_rate=(1, 1),
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         drop_prob=0.,
                         is_training=True):
    """
    Drop connect for 2D convolution layer

    drop_prob: drop rate for weight kernel
    is_training: bool, True for training, False for test
    """
    x = DropConnect(Conv2D(nb_filters,
                           kernel_size,
                           strides=strides,
                           padding=padding,
                           dilation_rate=dilation_rate,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint),
                    prob=drop_prob, training=is_training)(x)

    return x


def drop_connect_conv_3d(x,
                         nb_filters,
                         kernel_size,
                         strides=(1, 1, 1),
                         padding='valid',
                         dilation_rate=(1, 1, 1),
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         drop_prob=0.,
                         is_training=True):
    """
    Drop connect for 3D convolution layer

    drop_prob: drop rate for weight kernel
    is_training: bool, True for training, False for test
    """
    x = DropConnect(Conv3D(nb_filters,
                           kernel_size,
                           strides=strides,
                           padding=padding,
                           dilation_rate=dilation_rate,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint),
                    prob=drop_prob, training=is_training)(x)

    return x


def drop_connect_dense(x,
                         units,
                         activation=None,
                         use_bias=True,
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros',
                         kernel_regularizer=None,
                         bias_regularizer=None,
                         activity_regularizer=None,
                         kernel_constraint=None,
                         bias_constraint=None,
                         drop_prob=0.,
                         is_training=True):
    """
    Drop connect for densely-connected NN layer

    drop_prob: drop rate for weight kernel
    is_training: bool, True for training, False for test
    """
    x = DropConnect(Dense(units,
                           activation=activation,
                           use_bias=use_bias,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           bias_regularizer=bias_regularizer,
                           activity_regularizer=activity_regularizer,
                           kernel_constraint=kernel_constraint,
                           bias_constraint=bias_constraint),
                    prob=drop_prob, training=is_training)(x)

    return x