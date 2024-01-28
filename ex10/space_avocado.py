import pandas as pd
import numpy as np
from tqdm import tqdm

np.random.seed(21)

class LinearModel():
    def __init__(self, theta, alpha=0.01, max_iter=10) -> None:
        self.theta = theta
        self.alpha = alpha
        self.max_iter = max_iter

    def fit(self, x_, y):
        # def simple_grad(x_, theta):
        m = y.shape[0]
        x_one = np.hstack((np.ones((x_.shape[0], 1)), x_))
        self.theta = np.linalg.inv(x_one.T.dot(x_one)) @ x_one.T.dot(y)
        #     x_one_theta = x_one @ theta
        #     return x_one.T.dot(x_one_theta - y) / m
        # for _ in tqdm(range(self.max_iter)):
        #     self.theta = simple_grad(x, self.theta) * self.alpha
            


    def predict(self, x):
        
        x_one = np.hstack((np.ones((x.shape[0], 1)), x))
        # print(x_one.shape, self.theta.shape)
        return x_one @ self.theta
    
    def mse(self, y_hat, y):
        m = y.shape[0]
        return np.sum((y - y_hat)**2) / m 

class Polynom:
    def __init__(self, power):
        self.power = power

    def polynomial(self, x):
        if not isinstance(x, np.ndarray):
            return None
        return np.column_stack([x**(i+1) for i in range(self.power)])

def split_data(x, y, proportion=0.8):
    m = int(x.shape[0] * proportion)
    idxs = list(range(len(x)))
    np.random.shuffle(idxs)
    x = np.array([x[i] for i in idxs])
    y = np.array([y[i] for i in idxs])
    x_train, x_test, y_train, y_test = x[:m], x[m:], y[:m], y[m:]
    return x_train, x_test, y_train, y_test

def main():
    df = pd.read_csv("space_avocado.csv")
    y = df['target'].values[1:]
    x = df.iloc[:,1:-1].values[1:]
    x_train, x_test, y_train, y_test = split_data(x, y, 0.8)
    pl = Polynom(1)
    x_train = pl.polynomial(x_train)
    x_test = pl.polynomial(x_test)
    
    th = np.random.rand(x_train.shape[1]+1).reshape(x_train.shape[1]+1, -1)
    print(x_train.shape)
    model = LinearModel(th)
    model.fit(x_train, y_train)
    # print(x_test.shape, y_test.shape)
    y_hat = model.predict(x_test)
    
    # return y_hat.shape, y_test.shape
    return model.mse(y_hat, y_test)


if __name__=='__main__':
    
    print(main())