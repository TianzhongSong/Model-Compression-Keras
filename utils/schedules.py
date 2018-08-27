import keras.backend as K
from keras.callbacks import Callback


class Step(Callback):

    def __init__(self, steps, learning_rates, verbose=0):
        self.steps = steps
        self.lr = learning_rates
        self.verbose = verbose

    def change_lr(self, new_lr):
        old_lr = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose == 1:
            print('Learning rate is %g' %new_lr)

    def on_epoch_begin(self, epoch, logs={}):
        for i, step in enumerate(self.steps):
            if epoch < step:
                self.change_lr(self.lr[i])
                return
        self.change_lr(self.lr[i+1])

    def get_config(self):
        config = {'class': type(self).__name__,
                  'steps': self.steps,
                  'learning_rates': self.lr,
                  'verbose': self.verbose}
        return config

    @classmethod
    def from_config(cls, config):
        offset = config.get('epoch_offset', 0)
        steps = [step - offset for step in config['steps']]
        return cls(steps, config['learning_rates'],
                   verbose=config.get('verbose', 0))


def onetenth_120_160(lr):
    steps = [120, 160]
    lrs = [lr, lr/10, lr/100]
    return Step(steps, lrs)
