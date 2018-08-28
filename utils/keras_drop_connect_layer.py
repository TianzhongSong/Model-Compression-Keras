from keras.layers import Wrapper
import keras.backend as K


class DropConnect(Wrapper):
    def __init__(self, layer, prob=0., training=False, **kwargs):
        self.prob = prob
        self.training = training
        self.layer = layer
        super(DropConnect, self).__init__(layer, **kwargs)
        if self.training:
            if 0. < self.prob < 1.:
                self.uses_learning_phase = True

    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(DropConnect, self).build()

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def call(self, x):
        if self.training:
            if 0. < self.prob < 1.:
                self.layer.kernel = K.in_train_phase(K.dropout(self.layer.kernel, self.prob), self.layer.kernel)
                # self.layer.bias = K.in_train_phase(K.dropout(self.layer.bias, self.prob), self.layer.bias)
        return self.layer.call(x)
