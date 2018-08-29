from keras.layers import Input, MaxPool3D, Flatten
from keras.models import Model
from utils.keras_wrapped_layers import drop_connect_conv_3d, drop_connect_dense


def c3d_model(nb_classes, input_shape=(112, 112, 8, 3), is_training=True):
    model_inputs = Input(shape=input_shape)

    x = drop_connect_conv_3d(model_inputs, 64, (3, 3, 3),
                             padding='same', activation='relu', drop_prob=0.0, is_training=is_training)
    x = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = drop_connect_conv_3d(x, 128, (3, 3, 3), padding='same',
                             activation='relu', drop_prob=0.5, is_training=is_training)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = drop_connect_conv_3d(x, 128, (3, 3, 3), padding='same',
                             activation='relu', drop_prob=0.5, is_training=is_training)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = drop_connect_conv_3d(x, 256, (3, 3, 3), padding='same',
                             activation='relu', drop_prob=0.5, is_training=is_training)
    x = MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(x)

    x = drop_connect_conv_3d(x, 256, (3, 3, 3), padding='same',
                             activation='relu', drop_prob=0.5, is_training=is_training)
    x = MaxPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1), padding='same')(x)

    x = Flatten()(x)
    x = drop_connect_dense(x, 2048, activation='relu', drop_prob=0.5, is_training=is_training)
    x = drop_connect_dense(x, 2048, activation='relu', drop_prob=0.5, is_training=is_training)
    x = drop_connect_dense(x, nb_classes, activation='softmax', drop_prob=0.5, is_training=is_training)

    model = Model(inputs=model_inputs, outputs=x)
    return model
