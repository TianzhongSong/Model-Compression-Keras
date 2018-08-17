import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.path as path
import matplotlib.patches as patches
from tqdm import tqdm
import argparse


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
    # todo: find more efficient method for pruning threshold
    th = np.std(weight)
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
    indexes = np.argwhere(np.abs(weight) > th)
    values = weight[np.abs(weight) > th]
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
        else:
            ind = [index[0] * shape[1] + index[1]]
        if prev == 0:
            t = ind[0]
            if t > 255:
                t -= 255
                inds.append(255)
                vals.append(0.)
                while t > 255:
                    t -= 255
                    inds.append(255)
                    vals.append(0.)
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
                vals.append(0.)
            inds.append(t)
            vals.append(value)
        prev = ind[0]
    pbdr.close()
    vals = np.array(vals)
    inds = np.array(inds)
    vals, scale = quantize(vals)
    return inds, vals, scale, shape


def compression(weight_file, output_file):
    weights = h5py.File(weight_file, mode='r')
    compressed = h5py.File(output_file, mode='w')
    try:
        weights = weights['model_weights']
    except:
        pass
    try:
        layers = weights.attrs['layer_names']
    except:
        raise ValueError("hdf5 file must contain attribution: 'layer_names'")
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
                        indexes, values, scale, shape = prune(weight_value)
                        # plot_histograms(values)
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
                        pk = f.create_dataset(weight_name, shape, dtype='float32')
                        pk[:] = weight_value
                else:
                    weight_name = str(weight_name).split("'")[1].split('/')[-1]
                    pbias = f.create_dataset(weight_name, shape, dtype='float32')
                    pbias[:] = weight_value
    compressed.flush()
    compressed.close()
    print('Converting done')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--input-weight', type=str)
    parse.add_argument('--output-weight', type=str)
    args = parse.parse_args()
    compression(args.input_weight, args.output_weight)
