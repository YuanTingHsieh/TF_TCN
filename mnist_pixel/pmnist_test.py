from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

from .utils import data_generator
from .model import TCN

import argparse
import numpy as np
import tensorflow as tf
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--dropout', type=float, default=0.05,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=2e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

DATA_PATH = "/home/jimisv/yuanting/TF_TCN/mnist_pixel/MNIST_data/"
batch_size = args.batch_size
n_classes = 10
in_channels = 1
seq_length = int(784 / in_channels)
epochs = args.epochs
steps = 0

print(args)
X_train, Y_train, X_test, Y_test = data_generator(DATA_PATH)

labels = tf.placeholder(tf.int32, (batch_size, n_classes))
inputs = tf.placeholder(tf.float32, (batch_size, seq_length, in_channels))

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout
outputs = TCN(inputs, n_classes, channel_sizes, seq_length, kernel_size=kernel_size, dropout=dropout)
predictions = tf.argmax(outputs, axis=-1)
predictions_one_hot = tf.one_hot(predictions, depth=n_classes, axis=-1)

# outputs is of size (batch_size, n_classes)
# we only use the last element of sequence length
loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= outputs, labels=labels)

lr = args.lr
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
gradients, variables = zip(*optimizer.compute_gradients(loss))
if args.clip > 0:
    gradients, _ = tf.clip_by_global_norm(gradients, args.clip)
update_step = optimizer.apply_gradients(zip(gradients, variables))

def index_generator(num_examples, batch_size):
    all_indices = np.arange(num_examples)
    start_pos = 0
    #while True:
    #    all_indices = np.random.permutation(all_indices)
    for batch_idx, batch in enumerate(range(start_pos, num_examples, batch_size)):

        start_ind = batch
        end_ind = start_ind + batch_size

        # last batch
        if end_ind > num_examples:
            diff = end_ind - num_examples
            toreturn = all_indices[start_ind:end_ind]
            toreturn = np.append(toreturn, all_indices[0:diff])
            yield batch_idx + 1, toreturn
            start_pos = diff
            break

        yield batch_idx + 1, all_indices[start_ind:end_ind]

def train(ep, sess):
    global batch_size, seq_len, iters, epochs, total_steps
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    total_batches = len(X_train) // batch_size+1

    for batch_idx, indices in index_generator(len(X_train), batch_size):
        #print(batch_idx)
        x = X_train[indices]
        y = Y_train[indices]
        x = np.reshape(x, x.shape+ (1,))
        
        sess.run(update_step, feed_dict={inputs: x, labels: y})
        p, l = sess.run([predictions_one_hot, loss], feed_dict={inputs: x, labels: y})
        
        correct += np.sum(p == y)
        counter += p.size
        total_loss += l.mean()

        total_steps += 1
        '''
        if (batch_idx > 0 and batch_idx % args.log_interval == 0) or batch_idx == total_batches:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, len(X_train) // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0
        '''
        if (total_steps > 0 and total_steps % args.log_interval == 0):
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Steps {:5d} | lr {:2.5f} | ms/batch {:5.2f} | '
                  'train_loss {:5.8f} | train_accuracy {:5.4f}'.format(
                 total_steps, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            test()
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0

def test():
    global batch_size, seq_len, iters, epochs

    total_pred = np.zeros(Y_test.shape)
    total_loss = np.zeros(len(Y_test))
    for batch_idx, batch in enumerate(range(0, len(X_test), batch_size)):
        start_idx = batch
        end_idx = batch + batch_size

        x = X_test[start_idx:end_idx]
        y = Y_test[start_idx:end_idx]
        exclude = 0
        if len(x) < batch_size:
            exclude = batch_size - len(x)
            x = np.pad(x, ((0, exclude), (0, 0)), 'constant')
            y = np.pad(y, ((0, exclude), (0, 0)), 'constant')

        x = np.reshape(x, x.shape + (1,))

        p, l = sess.run([predictions_one_hot, loss], feed_dict={inputs: x, labels: y})

        if exclude > 0:
            total_pred[start_idx:end_idx] = p[:-exclude]
            total_loss[start_idx:end_idx] = l[:-exclude]
        else:
            total_pred[start_idx:end_idx] = p
            total_loss[start_idx:end_idx] = l

    print('| test_loss {:5.8f} | test_accuracy {:5.4f}'.format(total_loss.mean(), 100. * np.sum(p == y)/p.size ) )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    total_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total variables {:5d}'.format(total_variables))

    global total_steps
    total_steps = 0
    for ep in range(1, epochs + 1):
        train(ep, sess)
        # could add learning rate decay here as original example
    test()
