from pathlib import Path
print('Running' if __name__ == '__main__' else 'Importing', Path(__file__).resolve())

import numpy as np

def data_generator(N, seq_length):
    """
    Args:
        seq_length: Length of the adding problem data
        N: # of data in the set
    """
    X_num = np.random.rand(N*seq_length).reshape([N, seq_length, 1])
    X_mask = np.zeros([N, seq_length, 1])
    Y = np.zeros([N, 1])
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, positions[0], 0] = 1
        X_mask[i, positions[1], 0] = 1
        Y[i,0] = X_num[i, positions[0], 0] + X_num[i, positions[1], 0]
    X = np.concatenate((X_num, X_mask), 2)
    return X, Y