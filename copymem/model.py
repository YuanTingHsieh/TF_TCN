from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import tensorflow as tf
from tcn.tcn import TemporalConvNet

def TCN(input_layer, output_size, num_channels, sequence_length, kernel_size, dropout):
    tcn = TemporalConvNet(input_layer=input_layer, num_channels=num_channels, sequence_length=sequence_length, kernel_size=kernel_size, dropout=dropout)
    linear = tf.contrib.layers.fully_connected(tcn, output_size, activation_fn=None)
    return linear
