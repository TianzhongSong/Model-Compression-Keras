import numpy as np
import tensorflow as tf
import h5py
from tqdm import tqdm
from copy import deepcopy


def quantize(weights, bits=8):
    # not perfect, waiting for improve
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / ((2**(bits - 1)) - 1)
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def get_weights(model):
    """
    Get model weights
    """
    weights = {}
    for layer in model.layers:
        weight = layer.get_weights()
        if len(weight) > 0:
            weights[layer.name] = weight

    return weights


def set_weights(model, weights):
    dropped_weights = {}
    for i in range(len(model.layers)):
        if model.layers[i].name in weights:
            weight = deepcopy(weights[model.layers[i].name])
            if len(weight) > 0:
                if len(weight[0].shape) >= 2:
                    drop_rate = model.layers[i].prob
                    with tf.device('/cpu:0'):
                        x = tf.constant(weight[0], dtype=tf.float32)
                        weight_tensor = tf.nn.dropout(x, 1. - drop_rate)
                    with tf.Session() as sess:
                        tmp, s = quantize(sess.run(weight_tensor))
                        weight[0] = np.float32(tmp) * s
                        del x
                        del weight_tensor
                    model.layers[i].set_weights(weight)
                dropped_weights[model.layers[i].name] = weight
    return model, dropped_weights


def prune_weights(weight):
    if len(weight.shape) == 4:
        # pre-processing for 2D convolution
        weight = np.transpose(weight, (3, 2, 0, 1))
        shape = weight.shape
        shape_arr = np.array(list(shape))
        shape_arr = np.array([[np.prod(shape_arr[1:])], [np.prod(shape_arr[2:])], [shape_arr[2]]])
    elif len(weight.shape) == 5:
        # todo: pre-processing for 3D convolution
        weight = np.transpose(weight, (3, 2, 0, 1))
        shape = weight.shape
        shape_arr = np.array(list(shape))
        shape_arr = np.array([[np.prod(shape_arr[1:])], [np.prod(shape_arr[2:])], [shape_arr[2]]])
    else:
        # pre-processing for 1D convolution and fully-connected layer
        weight = np.transpose(weight, (1, 0))
        shape = weight.shape
        shape_arr = np.array(list(shape))[0]
    indexes = np.argwhere(np.abs(weight) > 0)
    values = weight[np.abs(weight) > 0]
    inds = []
    vals = []
    prev = 0
    print('Progress:')
    pbdr = tqdm(total=len(values))
    for index, value in zip(indexes, values):
        pbdr.update(1)
        # compute index
        if len(shape) == 4:
            ind = np.dot(index[:len(shape) - 1], shape_arr) + index[-1]
        elif len(shape) == 5:
            # todo
            pass
        elif len(shape) == 3:
            # todo
            pass
        else:
            ind = [index[0] * shape[1] + index[1]]
        if len(inds) == 0:
            t = ind[0]
            if t > 255:
                while t > 255:
                    t -= 255
                    inds.append(255)
                    vals.append(0)
                inds.append(t)
                vals.append(value)
            else:
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
    vals, scale = quantize(vals)
    return inds, vals, scale, shape


def save_pruned_weights(weights, save_name):
    compressed = h5py.File(save_name + '.h5', mode='w')
    for layer in weights:
        f = compressed.create_group(layer)
        weight = weights[layer]
        if len(weight[0].shape) >= 2:
            print('Start pruning {}'.format(layer))
            indexes, values, scale, shape = prune_weights(weight[0])
            pind = f.create_dataset('indexes', indexes.shape, dtype='uint8')
            pval = f.create_dataset('weights', values.shape, dtype='int8')
            psh = f.create_dataset('shape', np.array(shape).shape, dtype='int32')
            pscale = f.create_dataset('scale', shape=(1,), dtype='float32')
            pind[:] = indexes
            pval[:] = values
            psh[:] = np.array(shape)
            pscale[:] = np.array([scale])
            if len(weight) == 2:
                pbias = f.create_dataset('bias', weight[1].shape, dtype='float32')
                pbias[:] = weight[1]
        else:
            for n in range(len(weight)):
                pdata = f.create_dataset(str(n), weight[n].shape, dtype=weight[n].dtype)
                pdata[:] = weight[n]
    compressed.flush()
    compressed.close()
    print('Save {} done.'.format(save_name))


def select_best_model(model,x_test, y_test, iter=100):
    weights = get_weights(model)
    min_loss = 9999.
    dropped_weights = None
    for i in range(iter):
        model, tmp_weights = set_weights(model, weights)
        score = model.evaluate(x_test, y_test, verbose=0)
        print('loss: {0}, acc: {1}'.format(score[0], score[1]))
        if score[0] < min_loss:
            min_loss = score[0]
            dropped_weights = deepcopy(tmp_weights)
    print('The min loss is {}'.format(min_loss))
    return dropped_weights
