from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

from .utils import data_generator
from .model import TCN

import argparse
import numpy as np
import tensorflow as tf
import time

parser = argparse.ArgumentParser(description='Sequence Modeling - The Adding Problem')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.0)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=10,
                    help='upper epoch limit (default: 10)')
parser.add_argument('--ksize', type=int, default=6,
                    help='kernel size (default: 6)')
parser.add_argument('--levels', type=int, default=7,
                    help='# of levels (default: 7)')
parser.add_argument('--seq_len', type=int, default=200,
                    help='sequence length (default: 200)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4e-3,
                    help='initial learning rate (default: 4e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=27,
                    help='number of hidden units per layer (default: 27)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
args = parser.parse_args()

in_channels = 2
n_classes = 1
batch_size = args.batch_size
seq_length = args.seq_len
epochs = args.epochs
n_train = 50000
n_test = 1000
steps = 0

print(args)
print("Producing data...")
X_train, Y_train = data_generator(n_train, seq_length)
X_test, Y_test = data_generator(n_test, seq_length)

tf.set_random_seed(args.seed)
labels = tf.placeholder(tf.float32, (batch_size, n_classes))
inputs = tf.placeholder(tf.float32, (batch_size, seq_length, in_channels))

# Note: We use a very simple setting here (assuming all levels have the same # of channels.
channel_sizes = [args.nhid]*args.levels
kernel_size = args.ksize
dropout = args.dropout

outputs = TCN(inputs, n_classes, channel_sizes, seq_length, kernel_size=kernel_size, dropout=dropout)
# outputs is of size (batch_size, n_classes)
loss = tf.losses.mean_squared_error(labels= labels, predictions=outputs )
tf.summary.scalar('mse', loss)
merged = tf.summary.merge_all()

lr = args.lr
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
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

def train(epoch, sess):
    global batch_size, steps
    total_loss = 0
    start_time = time.time()

    for batch_idx, indices in index_generator(n_train, batch_size):
        x = X_train[indices]
        y = Y_train[indices]
        
        #sess.run(update_step, feed_dict={inputs: x, labels: y})
        summary, _, p, l = sess.run([merged, update_step, outputs, loss],
            feed_dict={inputs: x, labels: y})
        
        total_loss += l
        steps += 1
        train_writer.add_summary(summary, steps)

        if (batch_idx > 0 and batch_idx % args.log_interval == 0):
            avg_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| Epoch {:3d} | {:5d}/{:5d} batches | lr {:2.5f} | ms/batch {:5.2f} | '
                  'loss {:5.8f} |'.format(
                ep, batch_idx, n_train // batch_size+1, args.lr, elapsed * 1000 / args.log_interval,
                avg_loss))
            start_time = time.time()
            total_loss = 0


def evaluate(sess):
    global batch_size

    total_pred = np.zeros(Y_test.shape)
    total_loss = []
    for batch_idx, batch in enumerate(range(0, n_test, batch_size)):
        start_idx = batch
        end_idx = batch + batch_size

        x = X_test[start_idx:end_idx]
        y = Y_test[start_idx:end_idx]
        
        exclude = 0
        if len(x) < batch_size:
            exclude = batch_size - len(x)
            x = np.pad(x, ((0, exclude), (0, 0), (0, 0)), 'constant')
            y = np.pad(y, ((0, exclude), (0, 0)), 'constant')

        p, l = sess.run([outputs, loss], feed_dict={inputs: x, labels: y})

        if exclude > 0:
            total_pred[start_idx:end_idx] = p[:-exclude]
            #total_loss += l
        else:
            total_pred[start_idx:end_idx] = p
            total_loss.append(l)
    mse = np.mean(np.square(total_pred - Y_test))
    print('Test MSE Loss {:5.8f} '.format(mse))

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter('./log/train', sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    total_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print('Total variables {:5d}'.format(total_variables))

    for ep in range(1, epochs + 1):
        train(ep, sess)
        evaluate(sess)




