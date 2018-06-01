from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

# Imports
import numpy as np
import tensorflow as tf

from .nn import *
from tensorflow.contrib.framework.python.ops import add_arg_scope

@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rates, filter_size=[3], 
    stride=[1], pad='SAME', init_scale=1., init=False, ema=None, counters={}):
    name = get_name('weightnorm_conv', counters)
    with tf.variable_scope(name):
        # currently this part is never used
        if init:
            print("initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters], 
                tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0,1,2])
            x_init = tf.nn.convolution(x, V_norm, pad, stride, dilation_rates)
            #x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0,1])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init, trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init, trainable=True)
            x_init = tf.reshape(scale_init,[1,1,num_filters])*(x_init - tf.reshape(m_init,[1,1,num_filters]))
            # apply nonlinearity
            x_init = tf.nn.relu(x_init)
            return x_init

        else:
            # size of V is L, Cin, Cout
            V = tf.get_variable('V', filter_size+[int(x.get_shape()[-1]),num_filters], 
                tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                initializer=None, trainable=True)

            # size of input x is N, L, Cin

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1, 2])

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, dilation_rates), b)

            print(x.get_shape())

            # apply nonlinearity
            x = tf.nn.relu(x)

            return x

def TemporalBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters, dropout=0.0, init=False):

    keep_prob = 1.0 - dropout

    in_channels = input_layer.get_shape()[-1]

    # num_filters is the hidden units in TCN
    # which is the number of out channels
    conv1 = weightNormConvolution1d(input_layer, out_channels, [dilation_rate], [filter_size], [stride], counters=counters, init=init)
    #conv1 = layerNormConvolution1d(input_layer, out_channels, [dilation_rate], [filter_size], [stride], counters=counters)
    dropout1 = tf.nn.dropout(conv1, keep_prob)

    conv2 = weightNormConvolution1d(dropout1, out_channels, [dilation_rate], [filter_size], [stride], counters=counters, init=init)
    #conv2 = layerNormConvolution1d(dropout1, out_channels, [dilation_rate], [filter_size], [stride], counters=counters)
    dropout2 = tf.nn.dropout(conv2, keep_prob)

    # highway connetions or residual connection
    #highway = weightNormConvolution1d(input_layer, out_channels, [dilation_rate], [1], [stride], counters=counters, init=init) if in_channels != out_channels else None
    highway = None
    if in_channels != out_channels:
        W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]),out_channels], 
            tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
        b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32, initializer=None, trainable=True)
        highway = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
    else:
        print("no highway conv")

    res = input_layer if highway is None else highway

    return tf.nn.relu(dropout2 + res)

def TemporalConvNet(input_layer, num_channels, sequence_length, kernel_size=2, dropout=0.0, init=False):
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        print(i)
        dilation_size = 2 ** i
        #if i == 0:
        #    in_channels = num_inputs
        #    input_layer = tf.placeholder(tf.float32, shape=(None, sequence_length, in_channels))
        #else:
        #    in_channels = num_channels[i-1]
        out_channels = num_channels[i]
        input_layer = TemporalBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                 counters=counters, dropout=dropout, init=init)

    return input_layer