# Imports
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def temporal_padding(x, padding=(1, 1)):
    """Pads the middle dimension of a 3D tensor.
    # Arguments
        x: Tensor or variable.
        padding: Tuple of 2 integers, how many zeros to
            add at the start and end of dim 1.
    # Returns
        A padded 3D tensor.
    """
    assert len(padding) == 2
    pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
    return tf.pad(x, pattern)

def attentionBlock(x, counters, dropout):
    """self attention block
    # Arguments
        x: Tensor of shape [N, L, Cin]
        counters: to keep track of names
        dropout: add dropout after attention
    # Returns
        A tensor of shape [N, L, Cin]
    """

    k_size = x.get_shape()[-1].value
    v_size = x.get_shape()[-1].value

    name = get_name('attention_block', counters)
    with tf.variable_scope(name):
        # [N, L, k_size]
        key = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                              kernel_initializer=tf.random_normal_initializer(0, 0.01))
        key = tf.nn.dropout(key, 1.0 - dropout)
        # [N, L, k_size]
        query = tf.layers.dense(x, units=k_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        query = tf.nn.dropout(query, 1.0 - dropout)
        value = tf.layers.dense(x, units=v_size, activation=None, use_bias=False,
                                kernel_initializer=tf.random_normal_initializer(0, 0.01))
        value = tf.nn.dropout(value, 1.0 - dropout)

        logits = tf.matmul(query, key, transpose_b=True)
        logits = logits / np.sqrt(k_size)
        weights = tf.nn.softmax(logits, name="attention_weights")
        output = tf.matmul(weights, value)

    return output


@add_arg_scope
def weightNormConvolution1d(x, num_filters, dilation_rate, filter_size=3, stride=[1],
                            pad='VALID', init_scale=1., init=False, gated=False,
                            counters={}, reuse=False):
    """a dilated convolution with weight normalization (Salimans & Kingma 2016)
       Note that init part is NEVER used in our code
       It relates to the data-dependent init in original paper 
    # Arguments
        x: A tensor of shape [N, L, Cin]
        num_filters: number of convolution filters
        dilation_rate: dilation rate / holes
        filter_size: window / kernel width of each filter
        stride: stride in convolution
        gated: use gated linear units (Dauphin 2016) as activation
    # Returns
        A tensor of shape [N, L, num_filters]
    """
    name = get_name('weight_norm_conv1d', counters)
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # currently this part is never used
        if init:
            print("initializing weight norm")
            # data based initialization of parameters
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            V_norm = tf.nn.l2_normalize(V.initialized_value(), [0, 1])

            # pad x
            left_pad = dilation_rate * (filter_size - 1)
            x = temporal_padding(x, (left_pad, 0))
            x_init = tf.nn.convolution(x, V_norm, pad, stride, [dilation_rate])
            #x_init = tf.nn.conv2d(x, V_norm, [1]+stride+[1], pad)
            m_init, v_init = tf.nn.moments(x_init, [0, 1])
            scale_init = init_scale/tf.sqrt(v_init + 1e-8)
            g = tf.get_variable('g', dtype=tf.float32, initializer=scale_init,
                                trainable=True)
            b = tf.get_variable('b', dtype=tf.float32, initializer=-m_init*scale_init,
                                trainable=True)
            x_init = tf.reshape(scale_init, [1, 1, num_filters]) \
                                * (x_init - tf.reshape(m_init, [1, 1, num_filters]))
            # apply nonlinearity
            x_init = tf.nn.relu(x_init)
            return x_init

        else:
            # Gating mechanism (Dauphin 2016 LM with Gated Conv. Nets)
            if gated:
                num_filters = num_filters * 2

            # size of V is L, Cin, Cout
            V = tf.get_variable('V', [filter_size, int(x.get_shape()[-1]), num_filters],
                                tf.float32, tf.random_normal_initializer(0, 0.01),
                                trainable=True)
            g = tf.get_variable('g', shape=[num_filters], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.), trainable=True)
            b = tf.get_variable('b', shape=[num_filters], dtype=tf.float32,
                                initializer=None, trainable=True)

            # size of input x is N, L, Cin

            # use weight normalization (Salimans & Kingma, 2016)
            W = tf.reshape(g, [1, 1, num_filters]) * tf.nn.l2_normalize(V, [0, 1])

            # pad x for causal convolution
            left_pad = dilation_rate * (filter_size  - 1)
            x = temporal_padding(x, (left_pad, 0))

            # calculate convolutional layer output
            x = tf.nn.bias_add(tf.nn.convolution(x, W, pad, stride, [dilation_rate]), b)

            # GLU
            if gated:
                split0, split1 = tf.split(x, num_or_size_splits=2, axis=2)
                split1 = tf.sigmoid(split1)
                x = tf.multiply(split0, split1)
            # ReLU
            else:
                # apply nonlinearity
                x = tf.nn.relu(x)

            print(x.get_shape())

            return x

def TemporalBlock(input_layer, out_channels, filter_size, stride, dilation_rate, counters,
                  dropout, init=False, atten=False, use_highway=False, gated=False):
    """temporal block in TCN (Bai 2018)
    # Arguments
        input_layer: A tensor of shape [N, L, Cin]
        out_channels: output dimension
        filter_size: receptive field of a conv. filter
        stride: same as what's need in conv. function
        dilation_rate: holes inbetween
        counters: to keep track of layer names
        dropout: prob. to drop weights

        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation

        init: (NEVER used) data-dependent initialization

    # Returns
        A tensor of shape [N, L, out_channels]
    """
    keep_prob = 1.0 - dropout

    in_channels = input_layer.get_shape()[-1]
    name = get_name('temporal_block', counters)
    with tf.variable_scope(name):

        # num_filters is the hidden units in TCN
        # which is the number of out channels
        conv1 = weightNormConvolution1d(input_layer, out_channels, dilation_rate,
                                        filter_size, [stride], counters=counters,
                                        init=init, gated=gated)
        # set noise shape for spatial dropout
        # refer to https://colab.research.google.com/drive/1la33lW7FQV1RicpfzyLq9H0SH1VSD4LE#scrollTo=TcFQu3F0y-fy
        # shape should be [N, 1, C]
        noise_shape = (tf.shape(conv1)[0], tf.constant(1), tf.shape(conv1)[2])
        out1 = tf.nn.dropout(conv1, keep_prob, noise_shape)
        if atten:
            out1 = attentionBlock(out1, counters, dropout)

        conv2 = weightNormConvolution1d(out1, out_channels, dilation_rate, filter_size,
            [stride], counters=counters, init=init, gated=gated)
        out2 = tf.nn.dropout(conv2, keep_prob, noise_shape)
        if atten:
            out2 = attentionBlock(out2, counters, dropout)

        # highway connetions or residual connection
        residual = None
        if use_highway:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            H = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)

            W_t = tf.get_variable('W_t', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_t = tf.get_variable('b_t', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            T = tf.nn.bias_add(tf.nn.convolution(input_layer, W_t, 'SAME'), b_t)
            T = tf.nn.sigmoid(T)
            residual = H*T + input_layer * (1.0 - T)
        elif in_channels != out_channels:
            W_h = tf.get_variable('W_h', [1, int(input_layer.get_shape()[-1]), out_channels],
                                  tf.float32, tf.random_normal_initializer(0, 0.01), trainable=True)
            b_h = tf.get_variable('b_h', shape=[out_channels], dtype=tf.float32,
                                  initializer=None, trainable=True)
            residual = tf.nn.bias_add(tf.nn.convolution(input_layer, W_h, 'SAME'), b_h)
        else:
            print("no residual convolution")

        res = input_layer if residual is None else residual

        return tf.nn.relu(out2 + res)

def TemporalConvNet(input_layer, num_channels, sequence_length, kernel_size=2,
                    dropout=tf.constant(0.0, dtype=tf.float32), init=False,
                    atten=False, use_highway=False, use_gated=False):
    """A stacked dilated CNN architecture described in Bai 2018
    # Arguments
        input_layer: Tensor of shape [N, L, Cin]
        num_channels: # of filters for each CNN layer
        kernel_size: kernel for every CNN layer
        dropout: channel dropout after CNN

        atten: (not in TCN) add self attention block after Conv.
        use_highway: (not in TCN) use highway as residual connection
        gated: (not in TCN) use gated linear unit as activation

        init: (NEVER used) data-dependent initialization

    # Returns
        A tensor of shape [N, L, num_channels[-1]]
    """
    num_levels = len(num_channels)
    counters = {}
    for i in range(num_levels):
        print(i)
        dilation_size = 2 ** i
        out_channels = num_channels[i]
        input_layer = TemporalBlock(input_layer, out_channels, kernel_size, stride=1, dilation_rate=dilation_size,
                                 counters=counters, dropout=dropout, init=init, atten=atten, gated=use_gated)

    return input_layer
