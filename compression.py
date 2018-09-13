import numpy as np
import h5py
from copy import deepcopy
from tqdm import tqdm
import time


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def prune_weights(weight, compress_rate=0.9):
    for i in range(weight.shape[-1]):
        tmp = deepcopy(weight[..., i])
        tmp = np.abs(tmp)
        tmp = tmp[tmp >= 0]
        tmp = np.sort(np.array(tmp))
        # compute threshold
        th = tmp[int(tmp.shape[0] * compress_rate)]
        weight[..., i][np.abs(weight[..., i]) < th] = 0
    mask = deepcopy(weight)
    mask[mask != 0] = 1
    return weight, mask


def encode(weight):
    if len(weight.shape) == 4:
        # for 2D convolution
        weight = np.transpose(weight, (3, 2, 0, 1))
        shape = weight.shape
        shape_arr = np.array(list(shape))
        shape_arr = np.array([[np.prod(shape_arr[1:])], [np.prod(shape_arr[2:])], [shape_arr[2]]])
    elif len(weight.shape) == 5:
        # for 3D convolution
        weight = np.transpose(weight, (4, 3, 2, 0, 1))
        shape = weight.shape
        shape_arr = np.array(list(shape))
        shape_arr = np.array([[np.prod(shape_arr[1:])], [np.prod(shape_arr[2:])],
                              [np.prod(shape_arr[3:])], [shape_arr[3]]])
    elif len(weight.shape) == 3:
        # for 1D convolution
        weight = np.transpose(weight, (2, 0, 1))
        shape = weight.shape
        shape_arr = np.array(list(shape))
        shape_arr = np.array([[np.prod(shape_arr[1:])], [shape_arr[1]]])
    else:
        # for fully connected layer
        weight = np.transpose(weight, (1, 0))
        shape = weight.shape
        shape_arr = np.array(list(shape)[1])
    # prune zero weights
    indexes = np.argwhere(weight != 0)
    values = weight[weight != 0]
    inds = []
    vals = []
    prev = 0
    print('Progress:')
    pbdr = tqdm(total=len(values))
    for index, value in zip(indexes, values):
        pbdr.update(1)
        # compute absolute index
        ind = np.dot(index[:len(shape) - 1], shape_arr) + index[-1]
        # compute index difference
        if len(inds) == 0:
            t = ind[0]
            while t > 255:
                t -= 255
                inds.append(255)
                vals.append(0)
            inds.append(t)
            vals.append(value)
        else:
            t = ind[0] - prev
            while t > 255:
                t -= 255
                inds.append(255)
                vals.append(0)
            inds.append(t)
            vals.append(value)
        prev = ind[0]
    pbdr.close()
    inds = np.array(inds)
    vals = np.array(vals)
    # quantize fp32 weights to int8
    vals, scale = quantize(vals)
    return inds, vals, scale, shape


def decode(weights, indexes, shape, scale):
    decoded = np.zeros(shape, dtype='float32')
    weights = weights * scale
    if len(shape) == 4:
        # for 2D convolution
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
        # for 3D convolution
        pbdr = tqdm(total=len(weights))
        index = 0
        for i, j in zip(weights, indexes):
            pbdr.update(1)
            index += j
            x = index // np.prod(shape[1:])
            y = (index % np.prod(shape[1:])) // np.prod(shape[2:])
            h = ((index % np.prod(shape[1:])) % np.prod(shape[2:])) // np.prod(shape[3:])
            c = (((index % np.prod(shape[1:])) % np.prod(shape[2:])) % np.prod(shape[3:])) // shape[4]
            w = (((index % np.prod(shape[1:])) % np.prod(shape[2:])) % np.prod(shape[3:])) % shape[4]
            decoded[x, y, h, c, w] = i
        pbdr.close()
        decoded = np.transpose(decoded, (3, 4, 2, 1, 0))
    elif len(shape) == 3:
        # for 1D convolution
        pbdr = tqdm(total=len(weights))
        index = 0
        for i, j in zip(weights, indexes):
            pbdr.update(1)
            index += j
            x = index // np.prod(shape[1:])
            y = (index % np.prod(shape[1:])) // shape[2]
            h = ((index % np.prod(shape[1:])) % np.prod(shape[2:])) % shape[2]
            decoded[x, y, h] = i
        pbdr.close()
        decoded = np.transpose(decoded, (1, 2, 0))
    else:
        # for fully connected layer
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


def save_compressed_weights(model, save_name):
    compressed = h5py.File('{}.h5'.format(save_name), mode='w')
    start = time.time()
    for layer in model.layers:
        weight = layer.get_weights()
        if len(weight) > 0:
            f = compressed.create_group(layer.name)
            if len(weight[0].shape) >= 2:
                indexes, values, scale, shape = encode(weight[0])
                pind = f.create_dataset('indexes', indexes.shape, dtype='uint8')
                pval = f.create_dataset('weights', values.shape, dtype='int8')
                psh = f.create_dataset('shape', np.array(shape).shape, dtype='int32')
                pscale = f.create_dataset('scale', shape=(1,), dtype='float32')
                pind[:] = indexes
                pval[:] = values
                psh[:] = np.array(shape)
                pscale[:] = np.array([scale])
                if len(weight) == 2:
                    pbias = f.create_dataset('bias', shape=weight[1].shape, dtype='float32')
                    pbias[:] = weight[1]
            else:
                for i, w in enumerate(weight):
                    pv = f.create_dataset(str(i), shape=w.shape, dtype='float32')
                    pv[:] = w
    compressed.flush()
    compressed.close()
    print('Converting done! Time usage: {}'.format(time.time() - start))


def compression(weight_file, compress_rate=0.9):
    weights = h5py.File(weight_file, mode='r')
    compressed = h5py.File('compressed_weights.h5', mode='w')
    start = time.time()
    try:
        weights = weights['model_weights']
    except:
        pass
    try:
        layers = weights.attrs['layer_names']
    except:
        raise ValueError("hdf5 file must contain attribution: 'layer_names'")
    # not compress first convolution layer
    first_conv = True
    for layer_name in layers:
        g = weights[layer_name]
        if len(g.attrs['weight_names']) > 0:
            layer_name = str(layer_name).split("'")[1]
            f = compressed.create_group(layer_name)
            for weight_name in g.attrs['weight_names']:
                weight_value = g[weight_name].value
                shape = weight_value.shape
                if len(shape) >= 2:
                    if not first_conv:
                        print('Start pruning {}'.format(weight_name))
                        weight_value, _ = prune_weights(weight_value, compress_rate=compress_rate)
                        indexes, values, scale, shape = encode(weight_value)
                        pind = f.create_dataset('indexes', indexes.shape, dtype='uint8')
                        pval = f.create_dataset('weights', values.shape, dtype='int8')
                        psh = f.create_dataset('shape', np.array(shape).shape, dtype='int32')
                        pscale = f.create_dataset('scale', shape=(1,), dtype='float32')
                        pind[:] = indexes
                        pval[:] = values
                        psh[:] = np.array(shape)
                        pscale[:] = np.array([scale])
                    else:
                        first_conv = False
                        weight_name = str(weight_name).split("'")[1].split('/')[-1]
                        pv = f.create_dataset(weight_name, shape, dtype='float32')
                        pv[:] = weight_value
                else:
                    weight_name = str(weight_name).split("'")[1].split('/')[-1]
                    pbias = f.create_dataset(weight_name, shape, dtype='float32')
                    pbias[:] = weight_value
    compressed.flush()
    compressed.close()
    print(time.time() - start)
    print('Converting done')


if __name__ == '__main__':
    compression('./weights/LeNet.h5')
