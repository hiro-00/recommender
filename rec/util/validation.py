import numpy as np

def kfold(num_samples, n_fold, shuffle = True):
    indices = np.arange(num_samples)
    for test_index in _get_test_indices(len(indices), n_fold, shuffle):
        test_mask = np.ones(num_samples, dtype=np.bool)
        test_mask[test_index] = False
        train_index = indices[test_mask]
        yield train_index, test_index

def _get_test_indices(num_samples, n_fold, shuffle):
    indices = np.arange(num_samples)
    if shuffle:
        np.random.shuffle(indices)
    fold_size = len(indices) // n_fold
    start = 0
    for _ in range(n_fold):
        test_indices = indices[start : start + fold_size]
        yield test_indices
        start += fold_size


for train, test in kfold(10,5):
    print(train)
    print(test)
    print("--")