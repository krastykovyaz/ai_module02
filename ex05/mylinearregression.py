import numpy as np
from typing import Tuple


class MyLinearRegression:
    def __init__(self, theta, alpha=1.6e-4, max_iter=200000) -> None:
        try:
            if (not self.isvalid_arrays((theta,))
                or not isinstance(alpha, (int, float))
                or not isinstance(max_iter, int)
                or theta.shape[1] != 1):
                raise ValueError('Wrong argument')
            self.theta = theta
            self.alpha = alpha
            self.max_iter = max_iter
        except Exception as e:
            print(e)
            return None

    @staticmethod
    def isvalid_arrays(arrays: Tuple[np.ndarray]) -> bool:
        return all([isinstance(obj, np.ndarray) \
            and len(obj.shape) == 2 \
            and obj.size != 0 \
            for obj in arrays])

    def predict_(self, x):
        try:
            if not self.isvalid_arrays((x,)):
                return None
            if x.shape[1] + 1 != self.theta.shape[0]:
                return None
            x_one = np.hstack((np.ones((x.shape[0], 1)), x))
            return x_one @ self.theta
        except Exception as e:
                print(e)
                return None
    
    def loss_elem_(self, y, y_hat):
        try:
            if not self.isvalid_arrays((y, y_hat)) \
                or y.shape != y_hat.shape:
                return None
            return np.abs(y-y_hat)**2
        except Exception as e:
                print(e)
                return None
    
    def loss_(self, y, y_hat):
        try:
            if not self.isvalid_arrays((y, y_hat)) \
                    or y.shape != y_hat.shape:
                    return None
            m = y.shape[0]
            return np.sum((y-y_hat)**2) / (m * 2)
        except Exception as e:
                print(e)
                return None
    
    def fit_(self, x, y):
        # try:
        if not self.isvalid_arrays((x, y, self.theta)) \
            or x.shape[0] != y.shape[0] \
            or y.shape[1] != 1 \
            or self.theta.shape[1] != 1 or x.shape[1] + 1 != self.theta.size:
            return None
        # print(x.shape)
        def simple_grad(x_, theta):
            x_one = np.hstack((np.ones((x_.shape[0], 1)), x_)) # 200x3->200x4
            theta_x_one = x_one.dot(theta) # 200x4 @ 4x1->200x1
            m = y.shape[0]
            grad = x_one.T @ (theta_x_one - y) / m # 4x200 @ 200x1
            # print(x.shape, theta_x_one.shape)
            return grad # 4x1
        for _ in range(self.max_iter):
            g = simple_grad(x, self.theta)
            
            self.theta -= (self.alpha * g)
        # print(sef.theta)
        return self.theta
        # except Exception as e:
        #     print(e)
        #     return None
    
    def mse_(self, y, y_hat):
        try:
            if not self.isvalid_arrays((y, y_hat)) \
                or y.shape != y_hat.shape:
                return None
            m = y.shape[0]
            return np.sum((y_hat-y)**2)/m
        except Exception as e:
                print(e)
                return None



if __name__=='__main__':
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLinearRegression(np.array([[1.], [1.], [1.], [1.], [1]]))
    # Example 0:
    y_hat = mylr.predict_(X)
    # Output:
    assert np.allclose(y_hat, np.array([[8.], [48.], [323.]]))
    # Example 1:
    loss_e = mylr.loss_elem_(Y, y_hat)
    # Output:
    assert np.allclose(loss_e, np.array([[225.], [0.], [11025.]]))
    # Example 2:
    los = (mylr.loss_(Y, y_hat))
    # Output:
    assert np.allclose(los, 1875.0)
    # Example 3:
    mylr.alpha = 1.6e-4
    mylr.max_iter = 200000
    mylr.fit_(X, Y)
    th = mylr.theta
    # Output:
    print(th)
    np.array([[1.8188], [2.767], [-3.74], [1.392], [1.74]])
    # Example 4:
    y_hat = mylr.predict_(X)
    # Output:
    assert np.allclose(y_hat, np.array([[23.417], [47.489], [218.065]]))
    # Example 5:
    loss_e = mylr.loss_elem_(Y, y_hat)
    # Output:
    assert np.allclose(loss_e, np.array([[0.174], [0.260], [0.004]]), 0.1)
    # Example 6:
    los = mylr.loss_(Y, y_hat)
    # Output:
    print(los)
    assert np.allclose(los, 0.0732, 0.0001)