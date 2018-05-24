from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import argparse

import numpy as np
from .utils import data_generator
from .model import TCN
import tensorflow as tf

import time

parser = argparse.ArgumentParser(description='Sequence Modeling - Copying Memory Task')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip, -1 means no clip (default: 1.0)')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit (default: 50)')
parser.add_argument('--ksize', type=int, default=8,
                    help='kernel size (default: 8)')
parser.add_argument('--iters', type=int, default=100,
                    help='number of iters per epoch (default: 100)')
parser.add_argument('--levels', type=int, default=8,
                    help='# of levels (default: 8)')
parser.add_argument('--blank_len', type=int, default=1000, metavar='N',
                    help='The size of the blank (i.e. T) (default: 1000)')
parser.add_argument('--seq_len', type=int, default=10,
                    help='initial history size (default: 10)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='report interval (default: 50')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--optim', type=str, default='RMSprop',
                    help='optimizer to use (default: RMSprop)')
parser.add_argument('--nhid', type=int, default=10,
                    help='number of hidden units per layer (default: 10)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()


batch_size = args.batch_size
seq_len = args.seq_len    # The size to memorize
epochs = args.epochs
iters = args.iters
T = args.blank_len
n_steps = T + (2 * seq_len)
n_classes = 10  # Digits 0 - 9
n_train = 10000
n_test = 1000

print(args)
print("Preparing data...")
train_x, train_y = data_generator(T, seq_len, n_train)
test_x, test_y = data_generator(T, seq_len, n_test)

in_channels = 1

labels = tf.placeholder(tf.int32, (batch_size, T + 2*seq_len))
inputs = tf.placeholder(tf.float32, (batch_size, T + 2*seq_len, in_channels))

one_hot = tf.one_hot(labels, depth=n_classes, axis=-1)

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
dropout = args.dropout
# output is of size (batch_size, seq_len, n_classes)
outputs = TCN(inputs, n_classes, channel_sizes, seq_len, kernel_size, dropout=dropout)
predictions = tf.argmax(outputs, axis=-1)

loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits= outputs, labels=one_hot)

lr = args.lr
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr)
gradients, variables = zip(*optimizer.compute_gradients(loss))
if args.clip > 0:
    gradients, _ = tf.clip_by_global_norm(gradients, args.clip)
update_step = optimizer.apply_gradients(zip(gradients, variables))


def index_generator(n_train, batch_size):
    all_indices = np.arange(n_train)
    start_pos = 0
    #while True:
    #    all_indices = np.random.permutation(all_indices)
    for batch_idx, batch in enumerate(range(start_pos, n_train, batch_size)):

        start_ind = batch
        end_ind = start_ind + batch_size

        # last batch
        if end_ind > n_train:
            diff = end_ind - n_train
            toreturn = all_indices[start_ind:end_ind]
            toreturn = np.append(toreturn, all_indices[0:diff])
            yield batch_idx + 1, toreturn
            start_pos = diff
            break

        yield batch_idx + 1, all_indices[start_ind:end_ind]

def train(ep):
    global batch_size, seq_len, iters, epochs
    total_loss = 0
    start_time = time.time()
    correct = 0
    counter = 0
    total_batches = n_train // batch_size+1

    for batch_idx, indices in index_generator(n_train, batch_size):
        #print(batch_idx)
        x = train_x[indices]
        y = train_y[indices]
        x = np.reshape(x, x.shape+ (1,))
        
        sess.run(update_step, feed_dict={inputs: x, labels: y})
        p, l = sess.run([predictions, loss], feed_dict={inputs: x, labels: y})
        
        correct += np.sum(p == y)
        counter += p.size
        total_loss += l.mean()

        if (batch_idx > 0 and batch_idx % args.log_interval == 0) or batch_idx == total_batches:
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} | accuracy {:5.4f}'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss, 100. * correct / counter))
            start_time = time.time()
            total_loss = 0
            correct = 0
            counter = 0

def evaluate():
    global batch_size, seq_len, iters, epochs

    total_pred = np.zeros(test_y.shape)
    total_loss = np.zeros(test_y.shape)
    for batch_idx, batch in enumerate(range(0, n_test, batch_size)):
        start_idx = batch
        end_idx = batch + batch_size

        x = test_x[start_idx:end_idx]
        y = test_y[start_idx:end_idx]
        exclude = 0
        if len(x) < batch_size:
            exclude = batch_size - len(x)
            x = np.pad(x, ((0, exclude), (0, 0)), 'constant')
            y = np.pad(y, ((0, exclude), (0, 0)), 'constant')

        x = np.reshape(x, x.shape + (1,))

        p, l = sess.run([predictions, loss], feed_dict={inputs: x, labels: y})

        if exclude > 0:
            total_pred[start_idx:end_idx] = p[:-exclude]
            total_loss[start_idx:end_idx] = l[:-exclude]
        else:
            total_pred[start_idx:end_idx] = p
            total_loss[start_idx:end_idx] = l

    print('| Loss {:5.8f} | Accuracy {:5.4f}'.format(total_loss.mean(), 100. * np.sum(p == y)/p.size ) )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    total_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total variables {:5d}'.format(total_variables))

    for ep in range(1, epochs + 1):
        train(ep, sess)
    evaluate()
