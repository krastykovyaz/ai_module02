import numpy as np

def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) :
        return None
    np.random.seed(21)
    idxs = list(range(len(x)))
    np.random.shuffle(idxs)
    x = np.array([x[idx] for idx in idxs])
    y = np.array([y[idx] for idx in idxs])
    m = int(x.shape[0] * proportion)
    x_train, x_test, y_train, y_test = x[:m], x[m:], y[:m], y[m:]
    return x_train, x_test, y_train, y_test

if __name__=='__main__':
    x1 = np.array([1, 42, 300, 10, 59]).reshape((-1, 1))
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 1:
    data_spliter(x1, y, 0.8)
    # Output:
    np.array([ 1, 59, 42, 300]), np.array([10]), np.array([0, 0, 1, 0]), np.array([1])
    # Example 2:
    data_spliter(x1, y, 0.5)
    # Output:
    np.array([59, 10]), np.array([ 1, 300, 42]), np.array([0, 1]), np.array([0, 0, 1])
    x2 = np.array([[ 1, 42],
    [300, 10],
    [ 59, 1],
    [300, 59],
    [ 10, 42]])
    y = np.array([0, 1, 0, 1, 0]).reshape((-1, 1))
    # Example 3:
    data_spliter(x2, y, 0.8)
    # Output:
    (np.array([[ 10, 42], [300, 59], [ 59, 1], [300, 10]]), \
    np.array([[ 1, 42]]), \
    np.array([0, 1, 0, 1]), \
    np.array([0]))
    # Example 4:
    print(data_spliter(x2, y, 0.5))
    # Output:
    (np.array([[59, 1], [10, 42]]),
    np.array([[300, 10],[300, 59],[1, 42]]),
    np.array([0, 0]),
    np.array([1, 1, 0]))