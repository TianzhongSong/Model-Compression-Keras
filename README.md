# Model-Compression-Keras

cnn compression for keras

#### English is not my native language; please excuse typing errors.

The implementation for ["deep compression"](https://arxiv.org/abs/1510.00149)(only apply weight pruning and quantization).

This implementation is a little different from original mathod in the paper.

Pruning small weights for each convolution filter. The pruning threshold is set by handcraft.

### How to use

1. Train the model (this repo currently support normal-cnn, ResNet, Inception, Densenet, Unet, PointNet)

For CIFAR-10 and compression rate=0.8:

    python train_and_compress_cnn.py --model='vgg' --data='c10' --compress-rate=0.8

For CIFAR-100 and compression rate=0.8:

    python train_and_compress_cnn.py --model='vgg' --data='c100' --compress-rate=0.8
    
For UNet and compression rate=0.8:

    python train_and_compress_unet.py --compress-rate=0.8
    
For PointNet and compression rate=0.8:
    
    python train_and_compress_pointnet.py --compress-rate=0.8
    
2. Decode and evaluation:

    python decode_and_evaluate_cnn.py --model='vgg'
    
    python decode_and_evaluate_unet.py
    
    python decode_and_evaluate_pointnet.py

### Results

#### CIFAR-10

VGG like net  parameter:1.34M, val acc:85.16%, storage:5273KB

<table width="95%">
  <tr>
    <td colspan=4 align=center>VGG like net</td>
  </tr>
  <tr>
    <td align=center><b>compress rate</td>
    <td align=center>parameters</td>
    <td align=center>val acc</td>
    <td align=center>storage</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>0.9</td>
    <td align=center width="10%"><b>0.134M</td>
    <td align=center width="10%"><b>71.77%</td>
    <td align=center width="10%"><b>306KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.8</td>
    <td align=center width="10%"><b>0.268M</td>
    <td align=center width="10%"><b>84.23%</td>
    <td align=center width="10%"><b>565KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.7</td>
    <td align=center width="10%"><b>0.402M</td>
    <td align=center width="10%"><b>86.18%</td>
    <td align=center width="10%"><b>825KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.6</td>
    <td align=center width="10%"><b>0.536M</td>
    <td align=center width="10%"><b>86.98%</td>
    <td align=center width="10%"><b>1085KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.5</td>
    <td align=center width="10%"><b>0.67M</td>
    <td align=center width="10%"><b>86.54%</td>
    <td align=center width="10%"><b>1344KB</td>
  </tr>
</table>

ResNet-50  parameter:0.762M, val acc:93.79%, storage:3366KB
<table width="95%">
  <tr>
    <td colspan=4 align=center>ResNet-50</td>
  </tr>
  <tr>
    <td align=center><b>compress rate</td>
    <td align=center>parameters</td>
    <td align=center>val acc</td>
    <td align=center>storage</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>0.9</td>
    <td align=center width="10%"><b>0.076M</td>
    <td align=center width="10%"><b>90.06%</td>
    <td align=center width="10%"><b>410KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.8</td>
    <td align=center width="10%"><b>0.152M</td>
    <td align=center width="10%"><b>90.96%</td>
    <td align=center width="10%"><b>557KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.7</td>
    <td align=center width="10%"><b>0.229M</td>
    <td align=center width="10%"><b>90.72%</td>
    <td align=center width="10%"><b>703KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.6</td>
    <td align=center width="10%"><b>0.305M</td>
    <td align=center width="10%"><b>91.07%</td>
    <td align=center width="10%"><b>851KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.5</td>
    <td align=center width="10%"><b>0.381M</td>
    <td align=center width="10%"><b>90.50%</td>
    <td align=center width="10%"><b>996KB</td>
  </tr>
</table>

#### Person segmentation

Refered repo: [Person-Segmentation-Keras](https://github.com/TianzhongSong/Person-Segmentation-Keras)

Unet parameter:9.55M, val acc:98.46%, storage:37410KB
<table width="95%">
  <tr>
    <td colspan=4 align=center>Unet</td>
  </tr>
  <tr>
    <td align=center><b>compress rate</td>
    <td align=center>parameters</td>
    <td align=center>val acc</td>
    <td align=center>storage</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.95</td>
    <td align=center width="10%"><b>0.478M</td>
    <td align=center width="10%"><b>91.93%</td>
    <td align=center width="10%"><b>936KB</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>0.9</td>
    <td align=center width="10%"><b>0.955M</td>
    <td align=center width="10%"><b>98.11%</td>
    <td align=center width="10%"><b>1877KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.8</td>
    <td align=center width="10%"><b>1.91M</td>
    <td align=center width="10%"><b>98.45%</td>
    <td align=center width="10%"><b>3741KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.7</td>
    <td align=center width="10%"><b>2.87M</td>
    <td align=center width="10%"><b>98.50%</td>
    <td align=center width="10%"><b>5612KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.6</td>
    <td align=center width="10%"><b>3.82M</td>
    <td align=center width="10%"><b>98.51%</td>
    <td align=center width="10%"><b>7482KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.5</td>
    <td align=center width="10%"><b>4.78M</td>
    <td align=center width="10%"><b>98.53%</td>
    <td align=center width="10%"><b>9354KB</td>
  </tr>
</table>

#### PointNet

Refered repo:[PointNet-Keras](https://github.com/TianzhongSong/PointNet-Keras)

PointNet  parameter:3.49M, val acc:88.23%, storage:13772KB
<table width="95%">
  <tr>
    <td colspan=4 align=center>PointNet</td>
  </tr>
  <tr>
    <td align=center><b>compress rate</td>
    <td align=center>parameters</td>
    <td align=center>val acc</td>
    <td align=center>storage</td>
  </tr>
  <tr>
    <td align=center width="10%"><b>0.9</td>
    <td align=center width="10%"><b>0.349M</td>
    <td align=center width="10%"><b>83.80%</td>
    <td align=center width="10%"><b>921KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.8</td>
    <td align=center width="10%"><b>0.698M</td>
    <td align=center width="10%"><b>87.21%</td>
    <td align=center width="10%"><b>1597KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.7</td>
    <td align=center width="10%"><b>1.05M</td>
    <td align=center width="10%"><b>87.18%</td>
    <td align=center width="10%"><b>2269KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.6</td>
    <td align=center width="10%"><b>1.4M</td>
    <td align=center width="10%"><b>88.11%%</td>
    <td align=center width="10%"><b>2945KB</td>
  </tr>
    <tr>
    <td align=center width="10%"><b>0.5</td>
    <td align=center width="10%"><b>1.75M</td>
    <td align=center width="10%"><b>88.35%</td>
    <td align=center width="10%"><b>3607KB</td>
  </tr>
</table>
