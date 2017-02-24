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


def mask_rmse(ratings, predicted):
    valid_num = np.count_nonzero(ratings != -1)
    predicted[ratings == -1] = -1
    return np.sqrt(np.sum(np.square((ratings - predicted))) / valid_num)

def list_rmse(rating_list, predicted):
    rmse = 0
    for rating in rating_list:
        #print("{} {}".format(rating[2], predicted[rating[0]-1][rating[1]-1]))
        rmse += np.square(rating[2] - predicted[rating[0]-1][rating[1]-1])
    return np.sqrt(rmse / len(rating_list))