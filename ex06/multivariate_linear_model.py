import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
from ex05.mylinearregression import MyLinearRegression as MyLR
#Age: Age of the spacecraft.
#Thrust_powern: Power of engines in 10 km/s.
#Terameters: Distance that the spacecraft has travelled in terameters.
#Sell_price: This is the prices at which the custommer bought the spacecraft (in kiloeuros).


def plot_lr(x: np.ndarray, y: np.ndarray, y_hat: np.ndarray, xlabel: str, colors: tuple, lr_model: MyLR):
    plt.xlabel(f"x: {xlabel}")
    plt.ylabel("y: sell price (in keuros)")
    plt.grid()

    plt.plot(x, y, "o", color=colors[0], label="Sell price")
    plt.plot(x, y_hat, "o", color=colors[1], label="Predicted sell price", markersize=3)

    theta_str = ", ".join([f"{theta[0]:.3f}" for theta in lr_model.theta])
    theta_str = f"[{theta_str}]"
    mse = f"{lr_model.mse_(y, y_hat):.3f}"
    plt.title(f"$\\theta$: {theta_str}; MSE: {mse}")
    plt.legend()
    print(f"{xlabel}: theta_str={theta_str}; mse={mse}")
    plt.show()




def construct(x, y, l_reg, y_pred):
    try:
        config = (
            ("age (in years)", ("darkblue", "dodgerblue")),
            ("thrust power (in 10Km/s)", ("g", "lime")),
            ("distance (in Tmeters)", ("darkviolet", "violet")),
        )
        for xlabel, colors in config:
            plot_lr(x, y, y_pred, xlabel, colors, l_reg)
    except Exception as e:
        print(e)
        return None



if __name__=='__main__':
    try:
        data = pd.read_csv("spacecraft_data.csv")
    except Exception as e:
        exit(e)
    X = np.array(data[['Age']])
    Y = np.array(data[['Sell_price']])
    theta1 = np.array([[1000.0], [-1.0]])
    myLR_age = MyLR(theta=theta1, alpha = 2.5e-5, max_iter = 100000)
    myLR_age.fit_(X[:,0].reshape(-1,1), Y)
    y_pred = myLR_age.predict_(X[:,0].reshape(-1,1))
    construct(X, Y, myLR_age, y_pred)
    # Output
    res = myLR_age.mse_(y_pred, Y)
    assert np.allclose(res, 55736.86719, 0.1)
    X = np.array(data[['Age','Thrust_power','Terameters']])
    Y = np.array(data[['Sell_price']])
    theta2 = np.array([[1.0], [1.0], [1.0], [1.0]])
    myLR_age = MyLR(theta = theta2, alpha=5e-8, max_iter=500000)
    
    myLR_age.fit_(X, Y)
    y_pred = myLR_age.predict_(X)
    construct(X, Y, myLR_age, y_pred)
    
    print(myLR_age.theta)
    print(myLR_age.mse_(y_pred,Y))