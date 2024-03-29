import numpy as np

def predict_(x, theta):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimensions m * n.
    theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
    y_hat as a numpy.array, a vector of dimensions m * 1.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    None if x or theta is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
            return None

        if x.size == 0 or theta.size == 0:
            return None

        if len(theta.shape) != 2 or theta.shape[1] != 1:
            return None

        if len(x.shape) != 2 or x.shape[1] + 1 != theta.shape[0]:
            return None
        
        x_one = np.hstack((np.ones((x.shape[0], 1)), x)) # [4,3]->[4,4]
        return x_one @ theta # [4x4] @ [4x1]->[4x1]
    except Exception as e:
        print(e)
        return None

if __name__=='__main__':
    x = np.arange(1,13).reshape((4,-1)) # [4x3]
    # Example 1:
    theta1 = np.array([5, 0, 0, 0]).reshape((-1, 1)) # [4x1]
    # Ouput:
    assert np.allclose(predict_(x, theta1), \
                       np.array([[5.], [5.], [5.], [5.]]))
    # Do you understand why y_hat contains only 5’s here?
    # Example 2:
    theta2 = np.array([0, 1, 0, 0]).reshape((-1, 1))
    # Output:    
    assert np.allclose(predict_(x, theta2), \
                       np.array([[ 1.], [ 4.], [ 7.], [10.]]))
    # Do you understand why y_hat == x[:,0] here?
    # Example 3:
    theta3 = np.array([-1.5, 0.6, 2.3, 1.98]).reshape((-1, 1))
    # Output:    
    assert np.allclose(predict_(x, theta3), \
                       np.array([[ 9.64], [24.28], [38.92], [53.56]]))
    # Example 4:
    theta4 = np.array([-3, 1, 2, 3.5]).reshape((-1, 1))
    print(predict_(x, theta4))
    # Output:
    np.array([[12.5], [32. ], [51.5], [71. ]])
