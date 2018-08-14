import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from tqdm import tqdm


def plot_histograms(weight):
    fig, ax = plt.subplots()
    n, bins = np.histogram(weight, 50)
    left = np.array(bins[:-1])
    right = np.array(bins[1:])
    bottom = np.zeros(len(left))
    top = bottom + n

    XY = np.array([[left, left, right, right], [bottom, top, top, bottom]]).T

    barpath = path.Path.make_compound_path_from_polys(XY)

    patch = patches.PathPatch(barpath)
    ax.add_patch(patch)

    ax.set_xlim(left[0], right[-1])
    ax.set_ylim(bottom.min(), top.max())
    plt.show()


def quantize(weights):
    abs_weights = np.abs(weights)
    vmax = np.max(abs_weights)
    s = vmax / 127.
    qweights = weights / s
    qweights = np.round(qweights)
    qweights = qweights.astype(np.int8)
    return qweights, s


def prune(weight):
    th = np.std(weight)
    weight = np.transpose(weight, (3, 2, 1, 0))
    indexes = np.argwhere(np.abs(weight) > th)
    values = weight[np.abs(weight) > th]
    shape = weight.shape
    shape_arr = np.array(list(shape))
    shape_arr = np.array([[np.prod(shape_arr[1:])], [np.prod(shape_arr[2:])], [shape_arr[2]]])
    inds = None
    vals = None
    prev = 0
    print('Progress:')
    pbdr = tqdm(total=len(values))
    for index, value in zip(indexes, values):
        pbdr.update(1)
        ind = np.dot(index[:3], shape_arr) + index[-1]
        if inds is None:
            t = ind[0]
            if t > 255:
                t -= 255
                inds = np.array([255])
                vals = np.array([0.])
                while t > 255:
                    t -= 255
                    inds = np.hstack((inds, np.array([255])))
                    vals = np.hstack((vals, np.array([0.])))
                inds = np.hstack((inds, t))
                vals = np.hstack((vals, np.array(value)))
            else:
                inds = ind
                vals = np.array(value)
        else:
            t = ind[0] - prev
            while t > 255:
                t -= 255
                inds = np.hstack((inds, np.array([255])))
                vals = np.hstack((vals, np.array([0.])))
            inds = np.hstack((inds, np.array(t)))
            vals = np.hstack((vals, np.array([value])))
        prev = ind[0]
    pbdr.close()
    vals, scale = quantize(vals)
    return inds, vals, scale


def compression(weight_file):
    weights = h5py.File(weight_file, mode='r')
    compressed = h5py.File('compressed_weights.h5', mode='w')
    try:
        weights = weights['model_weights']
    except:
        pass
    try:
        layers = weights.attrs['layer_names']
    except:
        raise ValueError("hdf5 file must contain attribution: 'layer_names'")
    compressed.attrs['layer_names'] = [name for name in weights.attrs['layer_names']]
    for layer_name in layers:
        f = compressed.create_group(layer_name)
        g = weights[layer_name]
        f.attrs['weight_names'] = g.attrs['weight_names']
        for weight_name in g.attrs['weight_names']:
            weight_value = g[weight_name].value
            shape = weight_value.shape
            if len(shape) >= 2:
                print('Start pruning {}'.format(weight_name))
                indexes, values, scale = prune(weight_value)
                # plot_histograms(values)
                pind = f.create_dataset('index', indexes.shape, dtype='uint8')
                pval = f.create_dataset(weight_name, values.shape, dtype='int8')
                psh = f.create_dataset('shape', np.array(shape).shape, dtype='int32')
                pscale = f.create_dataset('scale', shape=(1,), dtype='float32')
                pind[:] = indexes
                pval[:] = values
                psh[:] = np.array(shape)
                pscale[:] = np.array([scale])
            else:
                pval = f.create_dataset(weight_name, weight_value.shape, dtype='float32')
                pval[:] = weight_value
    compressed.flush()
    compressed.close()
    weights.close()
    print('Converting done')


if __name__ == '__main__':
    compression('./weights/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
