import h5py
import numpy as np
from lenet import LeNet
from tqdm import tqdm
from utils.load_mnist import load_data


def decode(weights, indexes, shape, scale):
    decoded = np.zeros(shape, dtype='float32')
    weights = weights * scale
    if len(shape) == 4:
        pbdr = tqdm(total=len(weights))
        index = 0
        for i, j in zip(weights, indexes):
            pbdr.update(1)
            index += j
            x = index // np.prod(shape[1:])
            y = (index % np.prod(shape[1:])) // np.prod(shape[2:])
            h = ((index % np.prod(shape[1:])) % np.prod(shape[2:])) // shape[3]
            w = ((index % np.prod(shape[1:])) % np.prod(shape[2:])) % shape[3]
            decoded[x, y, h, w] = i
        pbdr.close()
        decoded = np.transpose(decoded, (2, 3, 1, 0))
    elif len(shape) == 5:
        # todo
        pass
    else:
        pbdr = tqdm(total=len(weights))
        index = 0
        for i, j in zip(weights, indexes):
            pbdr.update(1)
            index += j
            x = index // shape[1]
            y = index % shape[1]
            decoded[x, y] = i
        pbdr.close()
        decoded = np.transpose(decoded, (1, 0))
    return decoded


def decode_weights(weight_file):
    weights = {}
    f = h5py.File(weight_file, mode='r')
    for layer_name in f:
        print(layer_name)
        g = f[layer_name]
        weights[layer_name] = []
        weight_names = [n for n in g]
        if 'indexes' in weight_names:
            weight = g['weights'].value
            indexes = g['indexes'].value
            shape = g['shape'].value
            scale = g['scale'].value
            print('Decoding {}'.format(layer_name))
            weight_value = decode(weight, indexes, shape, scale)
            weights[layer_name].append(weight_value)
            for weight_name in weight_names:
                if weight_name not in ['weights', 'indexes', 'shape', 'scale']:
                    weight_value = g[weight_name].value
                    weights[layer_name].append(weight_value)
        else:
            for weight_name in weight_names:
                weight_value = g[weight_name].value
                weights[layer_name].append(weight_value)
    return weights


def load_weights(model, weights):
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = [w for w in weights[model.layers[i].name]]
            model.layers[i].set_weights(weight)
    return model


if __name__ == '__main__':
    model = LeNet()
    weights = decode_weights('./compressed_weights.h5')
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = [w for w in weights[model.layers[i].name]]
            model.layers[i].set_weights(weight)
    model.summary()

    x_train, y_train, x_test, y_test = load_data('./data/')
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    x_train /= 255
    x_test /= 255

    acc = 0.
    pbdr = tqdm(total=x_test.shape[0])
    for i in range(x_test.shape[0]):
        pbdr.update(1)
        x = x_test[i, :]
        x = np.expand_dims(x, axis=0)
        out = model.predict(x)
        if np.argmax(out[0]) == y_test[i]:
            acc += 1
    pbdr.close()
    print('acc:{}'.format(acc / 10000))
