import tensorflow as tf
from WN_Layers import convolution,Relu,lRelu,add_noise,globalAvgPool,dropout,dense,FullConv
# we are replicating the architechture in paper
#first add noise
#three convolutionlayer 3*3*96
#2*2 maxpool and dropout
#three 3*3*192 convolution
#2*2 maxpool and dropout
#3*3*192, two 1*1*192
#global average and dense
def ConvPool_CNN(x, drop_rate=0.5, is_test=False, init=False, use_WN=False,
               use_BN=False, use_MOBN=False):
    x = add_noise(x, is_test=is_test, name='gaussian_noise',noise_type='n')

    x = convolution(x, num_filters=96, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv1', activation=lRelu)

    x = convolution(x, num_filters=96, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv2', activation=lRelu)

    x = convolution(x, num_filters=96, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv3', activation=lRelu)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_1')
    x = dropout(x, drop_rate=drop_rate, is_test=is_test, name='drop1')

    x = convolution(x, num_filters=192, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv4', activation=lRelu)

    x = convolution(x, num_filters=192, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv5', activation=lRelu)

    x = convolution(x, num_filters=192, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name='conv6', activation=lRelu)

    x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='max_pool_2')
    x = dropout(x, drop_rate=drop_rate, is_test=is_test, name='drop2')

    x = convolution(x, num_filters=192, init=init, use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  pad='VALID', name='conv7', activation=lRelu)

    x = FullConv(x, num_units=192, activation=lRelu, init=init, use_WN=use_WN,
               use_BN=use_BN,
               use_MOBN=use_MOBN,
               is_test=is_test, name='FullConv1')

    x = FullConv(x, num_units=192, activation=lRelu, init=init, use_WN=use_WN,
               use_BN=use_BN,
               use_MOBN=use_MOBN,
               is_test=is_test, name='FullConv2')

    x = globalAvgPool(x, name='Globalavgpool1')

    x = dense(x, num_units=10, activation=None, init=init, use_WN=use_WN,
               use_BN=use_BN,
               use_MOBN=use_MOBN,
               is_test=is_test, name='output_dense')

    return x









