from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import numpy as np

def data_generator(T, mem_length, b_size):
    """
    Generate data for the copying memory task
    :param T: The total blank time length
    :param mem_length: The length of the memory to be recalled
    :param b_size: The batch size
    :return: Input and target data tensor
    """
    seq = np.random.randint(1, 9, size=(b_size, mem_length))
    zeros = np.zeros((b_size, T))
    marker = 9 * np.ones((b_size, mem_length + 1))
    placeholders = np.zeros((b_size, mem_length))

    x = np.concatenate((seq, zeros[:, :-1], marker), 1)
    y = np.concatenate((placeholders, zeros, seq), 1)

    return x, y