# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(data, labels):
  #x = tf.placeholder(tf.float32, shape=(None, 784), name='Input')
  #y = tf.placeholder(tf.int32, name='Label')
  input_layer = tf.reshape(data, [-1, 28, 28, 1])
  
  # Convolutional Layer #1
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  print(conv1.shape)

  # Pooling Layer #1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4)

  # Logits Layer
  logits = tf.layers.dense(inputs=dropout, units=10)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  return loss, conv1


mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

print(train_data.shape)
train_data = train_data[0:100, :]

train_labels = train_labels[0:100]

loss, conv1 = cnn_model_fn(train_data, train_labels)
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
  sess.run(init_g)
  sess.run(init_l)
  #sess = tf.Session()
  l, c_out = sess.run([loss, conv1])

  x = tf.placeholder(tf.float32, shape=(None, 784), name='Input')
  input_layer = tf.reshape(x, [-1, 28, 28, 1])
  conv_try = tf.nn.convolution(input = input_layer, filter=(5, 5, 1, 1), padding="same", dilation_rate=1)

  c_t = sess.run([conv_try], feed_dict={x: train_data})