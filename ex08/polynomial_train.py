import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class MyPolynomialFeatures:
    def __init__(self, degree: int):
        self.degree = degree

    def transform(self, X):
        try:
            if not isinstance(X, np.ndarray):
                return None
            return np.column_stack([X**(1+i) for i in range(self.degree)])
        except Exception as e:
            print(e)
            return None
        

class MyLinearRegression:
    def __init__(self, theta):
        self.theta = theta

    def fit(self, X, y):
        try:
            if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray) \
                or X.size == 0 or y.size == 0:
                return None
            x_one = np.hstack((np.ones((X.shape[0], 1)), X))
            self.theta = np.linalg.inv(x_one.T.dot(x_one)) @ x_one.T.dot(y)
        except Exception as e:
            print(e)
            return None

    def predict(self, X):
        try:
            if not isinstance(X, np.ndarray) or X.size == 0:
                return None
            X_extended = np.hstack((np.ones((X.shape[0], 1)), X))
            return X_extended @ self.theta
        except Exception as e:
            print(e)
            return None

if __name__=='__main__':
    # Read and load the dataset
    df = pd.read_csv("are_blue_pills_magics.csv")

    # Extract features and target
    X = df[['Micrograms']].values
    y = df['Score'].values

    # Degrees for polynomial features
    degrees = np.arange(1, 7)

    # List to store MSE scores
    mse_scores = []

    # Plot data points
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='Micrograms', y='Score', label='Data Points')

    theta4 = np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1)
    theta5 = np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1)
    theta6 = np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1)

    # Train models, evaluate, and plot
    for degree in degrees:
        # Create polynomial features
        poly = MyPolynomialFeatures(degree=degree)
        X_poly = poly.transform(X)
        
        theta = np.random.rand(degree).reshape(-1,1)
        # Train linear regression model
        if degree == 4:
            theta=theta4
        elif degree == 5:
            theta=theta5
        elif degree == 6:
            theta=theta6
        model = MyLinearRegression(theta)
        model.fit(X_poly, y)

        # Evaluate model
        y_pred = model.predict(X_poly)
        
        mse = np.sum((y_pred - y)**2) / len(y)
        mse_scores.append(mse)

        # Plot model
        X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        X_pred_poly = poly.transform(X_pred)
        y_pred_plot = model.predict(X_pred_poly)
        plt.plot(X_pred, y_pred_plot, label=f'Degree {degree}')

    # Plot settings
    plt.title('Polynomial Regression Models')
    plt.xlabel('Micrograms')
    plt.ylabel('Score')
    plt.legend()
    plt.show()

    # Print MSE scores
    for degree, mse in zip(degrees, mse_scores):
        print(f'Degree {degree}: MSE = {mse:.2f}')
