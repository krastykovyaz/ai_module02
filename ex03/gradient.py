import numpy as np

def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector (n +1) * 1.
    Return:
    The gradient as a numpy.array, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.array.
    None if x, y and theta do not have compatible dimensions.
    None if x, y or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or \
            not isinstance(y, np.ndarray) or \
                not isinstance(theta ,np.ndarray):
            return None
        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None

        if len(x.shape) != 2 or len(y.shape) != 2 or x.shape[0] != y.shape[0] or y.shape[1] != 1:
            return None

        # if len(theta.shape) != 2 or theta.shape[1] != 1 or theta.shape[0] != x.shape[1] + 1:
        #     return None
        x_one = np.hstack((np.ones((x.shape[0], 1)), x)) # [7x3]->[7x4]
        # grad = np.zeros(theta.shape) # 3x1
        m = x.shape[0]
        # for _ in range(x.shape[0]):
        #     x_one.T.dot(x_one.dot(theta) - y)
        theta_y = x_one.dot(theta) - y # [7x4] @ [4x1] -> [7x1] - [7x1]->[7x1]
        return x_one.T.dot(theta_y) / m # [7x4] -> [4x7] @ [7x4]->[4x1]
    except Exception as e:
        print(e)
        return None

if __name__=='__main__':
    x = np.array([
    [ -6, -7, -9],
    [ 13, -2, 14],
    [ -7, 14, -1],
    [ -8, -4, 6],
    [ -5, -9, 6],
    [ 1, -5, 11],
    [ 9, -11, 8]]) # [7x3]
    y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1)) # [7x1]
    theta1 = np.array([0, 3,0.5,-6]).reshape((-1, 1)) # [4x1]
    # Example 1:
    # Output:
    assert np.allclose(gradient(x, y, theta1), 
    np.array([[ -33.71428571], [ -37.35714286], [183.14285714], [-393.]]))
    # Example 2:
    theta2 = np.array([0,0,0,0]).reshape((-1, 1))
    print(gradient(x, y, theta2))
    # Output:
    np.array([[ -0.71428571], [ 0.85714286], [23.28571429], [-26.42857143]])