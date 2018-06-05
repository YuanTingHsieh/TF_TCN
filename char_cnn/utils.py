import unidecode
from collections import Counter
import os
from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())


import pickle
import numpy as np
from .ptb import ptb

def data_generator(args):
    #file, testfile, valfile = getattr(observations, args.dataset)('data/')
    if args.dataset == 'ptb':
        file, testfile, valfile = ptb('./data')
    file_len = len(file)
    valfile_len = len(valfile)
    testfile_len = len(testfile)
    corpus = Corpus(file + " " + valfile + " " + testfile)

    #############################################################
    # Use the following if you want to pickle the loaded data
    #
    # pickle_name = "{0}.corpus".format(args.dataset)
    # if os.path.exists(pickle_name):
    #     corpus = pickle.load(open(pickle_name, 'rb'))
    # else:
    #     corpus = Corpus(file + " " + valfile + " " + testfile)
    #     pickle.dump(corpus, open(pickle_name, 'wb'))
    #############################################################

    return file, file_len, valfile, valfile_len, testfile, testfile_len, corpus


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, string):
        self.dict = Dictionary()
        for c in string:
            self.dict.add_word(c)
        self.dict.prep_dict()


def char_tensor(corpus, string):
    tensor = np.zeros(len(string), dtype=np.int32)
    for i in range(len(string)):
        tensor[i] = corpus.dict.char2idx[string[i]]
    return tensor

def index_generator(n_data, batch_size):
    all_indices = np.arange(n_data)
    start_pos = 0
    #while True:
    #    all_indices = np.random.permutation(all_indices)
    for batch_idx, batch in enumerate(range(start_pos, n_data, batch_size)):

        start_ind = batch
        end_ind = start_ind + batch_size

        # last batch
        if end_ind > n_data:
            diff = end_ind - n_data
            toreturn = all_indices[start_ind:end_ind]
            toreturn = np.append(toreturn, all_indices[0:diff])
            yield batch_idx + 1, toreturn
            start_pos = diff
            break

        yield batch_idx + 1, all_indices[start_ind:end_ind]

def batchify(data, batch_size):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Some data cross the boundary is discarded
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = len(data) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch*batch_size]
    # Evenly divide the data across the batch_size batches.
    data = data.reshape((batch_size, -1))
    return data

def get_batch(source, start_index, args):
    seq_len = min(args.seq_len, source.shape[1] - 1 - start_index)
    end_index = start_index + seq_len
    inp = source[:, start_index:end_index]
    target = source[:, start_index+1:end_index+1]  # The successors of the inp.
    return inp, target


def save(model):
    #save_filename = 'model.pt'
    #torch.save(model, save_filename)
    print('Dummy NOT Saved as %s' % save_filename)


