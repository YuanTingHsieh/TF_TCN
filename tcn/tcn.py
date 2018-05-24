from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# Imports
import numpy as np
import tensorflow as tf

from .nn import *
from tensorflow.contrib.framework.python.ops import add_arg_scope

@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rates, filter_size=[3], stride=[1], pad='SAME', init_scale=1., init=False, ema=None, counters={}):
    name = get_name('weightnorm_conv', counters)
    with tf.variable_scope(name):
        V = get_var_maybe_avg('V', ema, shape=filter_size+[int(x.get_shape()[-1]),num_filters], dtype=tf.float32,
                              initializer=tf.random_normal_initializer(0, 0.01), trainable=True)
        g = get_var_maybe_avg('g', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg('b', ema, shape=[num_filters], dtype=tf.float32,
                              initializer=tf.constant_initializer(0.), trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, dilation_rates), b)

        #print(x.get_shape())

        if init:  # normalize x
            m_init, v_init = tf.nn.moments(x, [0,1,2])
            scale_init = init_scale / tf.sqrt(v_init + 1e-10)
            with tf.control_dependencies([g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
                x = tf.identity(x)

        # apply nonlinearity
        x = tf.nn.relu(x)

        return x

def TemporalBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters, dropout=0.0):

    keep_prob = 1.0 - dropout

    #self.input = tf.placeholder(tf.float32, shape=(None, sequence_length, in_channels))
    in_channels = input_layer.get_shape()[-1]

    # num_filters is the hidden units in TCN
    # which is the number of out channels
    conv1 = weightNormConvolution1d(input_layer, out_channels, [dilation_rate], [filter_size], [stride], counters=counters)
    dropout1 = tf.nn.dropout(conv1, keep_prob)

    conv2 = weightNormConvolution1d(dropout1, out_channels, [dilation_rate], [filter_size], [stride], counters=counters)
    dropout2 = tf.nn.dropout(conv2, keep_prob)

    # highway connetions or residual connection
    highway = weightNormConvolution1d(input_layer, out_channels, [dilation_rate], [1], [stride], counters=counters) if in_channels != out_channels else None
    
    res = input_layer if highway is None else highway

    return tf.nn.relu(dropout2 + res)

def TemporalConvNet(input_layer, num_channels, sequence_length, kernel_size=2, dropout=0.2):       
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        #print(i)
        dilation_size = 2 ** i
        #if i == 0:
        #    in_channels = num_inputs
        #    input_layer = tf.placeholder(tf.float32, shape=(None, sequence_length, in_channels))
        #else:
        #    in_channels = num_channels[i-1]
        out_channels = num_channels[i]
        input_layer = TemporalBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                 counters=counters, dropout=dropout)

    return input_layer