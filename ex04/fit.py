import numpy as np

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
    Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.array, a matrix of dimension m * n:
    (number of training examples, number of features).
    y: has to be a numpy.array, a vector of dimension m * 1:
    (number of training examples, 1).
    theta: has to be a numpy.array, a vector of dimension (n + 1) * 1:
    (number of features + 1, 1).
    alpha: has to be a float, the learning rate
    max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
    new_theta: numpy.array, a vector of dimension (number of features + 1, 1).
    None if there is a matching dimension problem.
    None if x, y, theta, alpha or max_iter is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
            return None

        if x.size == 0 or y.size == 0 or theta.size == 0:
            return None

        if len(x.shape) != 2 or len(y.shape) != 2 or y.shape[1] != 1 or x.shape[0] != y.shape[0]:
            return None

        if len(theta.shape) != 2 or theta.shape[1] != 1 or theta.shape[0] != x.shape[1] + 1:
            return None

        if not isinstance(alpha, (int, float)) or not isinstance(max_iter, int):
            return None

        if alpha <= 0 or max_iter <= 0:
            return None
        
        for _ in range(max_iter):
            x_one = np.hstack((np.ones((x.shape[0], 1)),x)) # [4x3]->[4x4]
            alpha_theta = np.dot(x_one, theta) # 4x4 @ 4x1-> 4x1
            grad = x_one.T.dot(alpha_theta - y) / len(y) # 4x4 @ 4x1 -> 4x1
            theta -= alpha * grad
        return theta
    except Exception as e:
        print(e)
        return None
    
def predict_(x, theta):
    try:
        x_one = np.hstack((np.ones((x.shape[0], 1)),x)) # [4x3]->[4x4]
        return x_one @ theta
    except Exception as e:
        print(e)
        return None



if __name__=='__main__':
    x = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    # Example 0:
    theta2 = fit_(x, y, theta, alpha = 0.0005, max_iter=42)
    # Output:
    print(theta2)
    np.array([[41.99], [0.97], [0.77], [-1.20]])
    # Example 1:
    print(predict_(x, theta2))
    # Output:
    np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]])