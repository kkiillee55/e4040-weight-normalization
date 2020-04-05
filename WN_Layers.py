import numpy as np
import tensorflow as tf

def input_size(x):
    # calculate the size of input
    return list(map(int, x.get_shape()))

#batch normalization
def BN(x,is_from_conv=True,is_test=False,decay=0.9,name='BN'):
    with tf.variable_scope(name):
        # the formula is (x-u)/std*gamma+beta, u is mean
        gamma = tf.get_variable('gamma', shape=x.get_shape()[-1], dtype=tf.float32, initializer=tf.ones_initializer(),trainable=True)
        beta = tf.get_variable('beta', shape=x.get_shape()[-1], dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=True)
        moving_mean = tf.get_variable('moving_mean', shape=x.get_shape()[-1], dtype=tf.float32,initializer=tf.zeros_initializer(), trainable=False)
        moving_var = tf.get_variable('moving_var', shape=x.get_shape()[-1], dtype=tf.float32,initializer=tf.ones_initializer(), trainable=False)
    if is_test:
        #during the test phase, we use moving mean and moving var which are calculated in training, won't change in testing
        return tf.nn.batch_normalization(x,moving_mean,moving_var,beta,gamma,0.001)
    else:
        #if the input data of bn is from the convolution layer, it's a 4D vector, calculate mean and var along 0,1,2 axis
        if is_from_conv:
            curr_mean,curr_var=tf.nn.moments(x,[0,1,2])
        else:
            #if the input data of bn is from the dense layer, it's a 1D vector, calculate mean and var along 0 axis
            curr_mean,curr_var=tf.nn.moments(x,[0])
        #the moving mean and moving var are the weighted sum of their own and current values
        moving_mean=tf.assign(moving_mean,decay*moving_mean+(1-decay)*curr_mean)
        moving_var=tf.assign(moving_var,decay*moving_var+(1-decay)*moving_var)
        # make sure to update moving mean and moving var before bn
        with tf.control_dependencies([moving_mean,moving_var]):
            return tf.nn.batch_normalization(x,curr_mean,curr_var,beta,gamma,0.001)

#mean only batch normalization
def MOBN(x,moving_mean,beta,is_from_conv=True,is_test=False,decay=0.9,name='MOBN'):
    with tf.variable_scope(name):
        if is_test:
            #if in test phase, we use moving mean which is calculated from training phase
            return x-moving_mean+beta
        else:
            #if the input of mobn is from the convolution layer, calculate mean along 0,1,2 axis
            if is_from_conv:
                curr_mean,_=tf.nn.moments(x,[0,1,2])
            else:
            # if the input is from dense layer, calcuate mean along 0 axis
                curr_mean,_=tf.nn.moments(x,[0])
            #calculate the moving mean which is the weight sum of its value and current mean
            moving_mean=tf.assign(moving_mean,decay*moving_mean+(1-decay)*curr_mean)
            #make sure to update the moving mean before mobn
            with tf.control_dependencies([moving_mean]):
                return x-curr_mean+beta

# the randomly mask neurons, the default drop rate is 50%, druing test, the drop rate is 0
def dropout(x, drop_rate=0.5, is_test=False, name=''):
    with tf.variable_scope(name):
        if is_test:
            return x
        else:
            x = tf.nn.dropout(x,rate=drop_rate,name=name)
            return x

def convolution(x, num_filters, filter_size=[3,3],  pad='SAME', stride=[1,1], activation=None, init_scale=1., init=False, use_WN=False, use_BN=False, use_MOBN=False,is_test=False,name=''):
    with tf.variable_scope(name):
        #the weight is parameterized as v/||v||*g,v has same shape as w, we first initialize v
        v=tf.get_variable('v',shape=filter_size+[int(x.get_shape()[-1]),num_filters],dtype=tf.float32,initializer=tf.random_normal_initializer(0,0.05),trainable=True)
        #get the moivng mean if we use mobn in convolution layer
        if use_MOBN:
            moving_mean=tf.get_variable('MOBN/moving_mean',shape=[num_filters],dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
        #initialize the bias
        if not use_BN:
            b=tf.get_variable('b',shape=[num_filters],dtype=tf.float32,initializer=tf.constant_initializer(0.),trainable=True)
        # g is also a trainable parameter is we use weight normalization
        if use_WN:
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,initializer=tf.constant_initializer(1.), trainable=True)
            #initialize weight normalization
            if init:
                #calculate the norm of v
                v_norm=tf.nn.l2_normalize(v,[0,1,2])
                x=tf.nn.conv2d(x,v_norm,strides=[1] + stride + [1],padding=pad)
                #calculate the mean and var of convolution, amke sure the features have zero mean and unit variance
                mean_init,var_init=tf.nn.moments(x,[0,1,2])
                scale_init = init_scale / tf.sqrt(var_init + 1e-08)
                assign_g = g.assign(scale_init)
                assign_b = b.assign(-mean_init*scale_init)
                with tf.control_dependencies([assign_g, assign_b]):
                    x = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x - tf.reshape(mean_init, [1, 1, 1, num_filters]))
            else:
                # if no initialization, just use w
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(v, [0, 1, 2])
                if use_MOBN:
                    # use weight-normalization and mobn
                    x = tf.nn.conv2d(x,W,strides=[1]+stride+[1],padding=pad)
                    x = MOBN(x,moving_mean,b,is_from_conv=True, is_test=is_test)
                else:
                    # only weight-normalization
                    x = tf.nn.bias_add(tf.nn.conv2d(x, W, [1] + stride + [1], pad), b)
        elif use_BN:
            # use bn
            x = tf.nn.conv2d(x,v,[1]+stride+[1],pad)
            x = BN(x,is_from_conv=True,is_test=is_test)
        else:
            x = tf.nn.bias_add(tf.nn.conv2d(x,v,strides=[1]+stride+[1],padding=pad),b)
        # the activation function of convolution
        if activation is not None:
            x = activation(x)
        return x

#for future VAE model, same structure as convolution
def convolution_transpose(x,num_filters, filter_size=[3,3],output_shape=None,  pad='SAME', stride=[1,1], activation=None, init_scale=1., init=False, use_WN=False, use_BN=False, use_MOBN=False,is_test=False,name=''):

    with tf.variable_scope(name):

        v=tf.get_variable('v',shape=filter_size+[int(x.get_shape()[-1]),num_filters],dtype=tf.float32,initializer=tf.random_normal_initializer(),trainable=True)
        if not use_BN:
            b=tf.get_variable('b',shape=[num_filters],dtype=tf.float32,initializer=tf.constant_initializer(0.),trainable=True)
        if use_MOBN:
            moving_mean=tf.get_variable('MOBN/moving_mean',shape=[num_filters],dtype=tf.float32, initializer=tf.zeros_initializer(),trainable=False)
        if use_WN:
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,initializer=tf.constant_initializer(1.), trainable=True)
            if init:
                v_norm=tf.nn.l2_normalize(v,[0,1,2])
                x=tf.nn.conv2d_transpose(x,v_norm,output_shape=output_shape,strides=[1] + stride + [1],padding=pad)
                mean_init,var_init=tf.nn.moments(x,[0,1,2])
                scale_init = init_scale / tf.sqrt(var_init + 1e-08)
                assign_g = g.assign(scale_init)
                assign_b = b.assign(-mean_init*scale_init)
                with tf.control_dependencies([assign_b,assign_g]):
                    x = tf.reshape(scale_init, [1, 1, 1, num_filters]) * (x - tf.reshape(mean_init, [1, 1, 1, num_filters]))
            else:
                W = tf.reshape(g, [1, 1, 1, num_filters]) * tf.nn.l2_normalize(v, [0, 1, 2])
                if use_MOBN: # use weight-normalization combined with mean-only-batch-normalization
                    x = tf.nn.conv2d_transpose(x,W,output_shape=output_shape,strides=[1]+stride+[1],padding=pad)
                    x = MOBN(x,moving_mean,b,is_from_conv=True, is_test=is_test)
                else:
                    # use just weight-normalization
                    x = tf.nn.bias_add(tf.nn.conv2d_transpose(x, W,output_shape=output_shape,strides= [1] + stride + [1], padding=pad), b)
        elif use_BN:
            x = tf.nn.conv2d_transpose(x,v,output_shape=output_shape,strides=[1]+stride+[1],padding=pad)
            x = BN(x,is_from_conv=True,is_test=is_test)
        else:
            x = tf.nn.bias_add(tf.nn.conv2d_transpose(x,v,output_shape=output_shape,strides=[1]+stride+[1],padding=pad),b)

        if activation is not None:
            x = activation(x)
        return x

# three activation function Relu, softmax and LeakyRelu
def Relu(x,name='Relu'):
    with tf.variable_scope(name):
        return tf.nn.relu(x)
def softmax(x,name='softmax'):
    with tf.variable_scope(name):
        return tf.nn.softmax(x)

def lRelu(x, alpha=0.1, name='leakyRelu'):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x,alpha=alpha)

#adding noise at the begnning, n is normal distributed noise, u is uniform distributed noise
def add_noise(x, sigma=0.15, maxval=0.5,is_test=False, name='',noise_type='n'):
    with tf.variable_scope(name):
        if is_test:
            return x
        else:
            if(noise_type=='n'):
                noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)
            if(noise_type=='u'):
                noise = tf.random_uniform(shape=tf.shape(x),maxval=maxval,dtype=tf.float32)
            return x + noise

# gloabl average pooling for input minibatch with size [batchsize,height,width, channles]
# calculate average on height and width axis.
def globalAvgPool(x, name=''):
    with tf.variable_scope(name):
        return tf.reduce_mean(x, axis=[1,2])



#dense layer after convolution, can also use bn, wn and mobn
def dense(x, num_units, activation=None, init_scale=1., init=False,
          use_WN=False, use_BN=False,
          use_MOBN=False,
          is_test=False, name=''):
    with tf.variable_scope(name):
        #same as convolution, initialize v, b,and g
        v = tf.get_variable('v', shape=[int(x.get_shape()[1]), num_units], dtype=tf.float32,initializer=tf.random_normal_initializer(0, 0.05), trainable=True)
        if use_BN is False:
            b = tf.get_variable('b', shape=[num_units], dtype=tf.float32,initializer=tf.constant_initializer(0.), trainable=True)
        if use_MOBN:
            moving_mean = tf.get_variable('MOBN/moving_mean', shape=[num_units], dtype=tf.float32,initializer=tf.zeros_initializer(), trainable=False)
        if use_WN:
            g = tf.get_variable('g', shape=[num_units], dtype=tf.float32,initializer=tf.constant_initializer(1.), trainable=True)
            if init:
                #initialize g and b, same as convolution layer
                v_norm = tf.nn.l2_normalize(v, [0])
                x = tf.matmul(x, v_norm)
                m_init, v_init = tf.nn.moments(x, [0])
                scale_init = init_scale / tf.sqrt(v_init + 1e-10)
                assign_g = g.assign(scale_init)
                assign_b = b.assign(-m_init * scale_init)
                with tf.control_dependencies([assign_g,assign_b]):
                    x = tf.reshape(scale_init, [1, num_units]) * (x - tf.reshape(m_init, [1, num_units]))
            else:

                x = tf.matmul(x, v)
                scaler = g / tf.sqrt(tf.reduce_sum(tf.square(v), [0]))
                x = tf.reshape(scaler, [1, num_units]) * x
                b = tf.reshape(b, [1, num_units])
                if use_MOBN:
                    x = MOBN(x, moving_mean, b, is_from_conv=False, is_test=is_test)
                else:
                    x = x + b
        #use bn
        elif use_BN:
            x = tf.matmul(x, v)
            x = BN(x, is_from_conv=False, is_test=is_test)
        #use no normalization
        else:
            x = tf.nn.bias_add(tf.matmul(x, v), b)
        # apply nonlinear activation
        if activation is not None:
            x = activation(x)

        return x

# 1*1 convolution, which is similar to dense layer, so rehape it and feed into dense layer , then reshape it back
def FullConv(x, num_units, activation=None, init=False, use_WN=False, use_BN=False,
        use_MOBN=False, is_test=False, name=''):

    with tf.variable_scope(name):
        s = input_size(x)
        x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
        x = dense(x, num_units=num_units, activation=activation, init=init,
                  use_WN=use_WN,
                  use_BN=use_BN,
                  use_MOBN=use_MOBN,
                  is_test=is_test,
                  name=name)
        return tf.reshape(x, s[:-1] + [num_units])


