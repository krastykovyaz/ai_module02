import pandas as pd
import numpy as np
from space_avocado import split_data
from space_avocado import Polynom
from space_avocado import LinearModel
import matplotlib.pyplot as plt
import yaml, dill

# Load and Explore the Dataset
df = pd.read_csv("space_avocado.csv")
# Explore the dataset (optional)

# Split the Dataset
y = df['target'].values[1:]
x = df.iloc[:,1:-1].values[1:]
X_train, X_test, y_train, y_test = split_data(x, y)
# X_train, X_test, y_train, y_test = X_train.values[1:], X_test.values[1:], y_train.values[1:].reshape(-1, 1), y_test.values[1:].reshape(-1, 1)
# Feature Engineering: Implement your polynomial_features method here
# def polynomial_features(x, degree):
    # Your implementation for polynomial features
    # return np.column_stack([x**(i+1) for i in range(degree)])

# Train Polynomial Regression Models
degrees = [1,2,3,4]
models = {}


for degree in degrees:
    pl = Polynom(degree)
    # Create polynomial features
    X_train_poly = pl.polynomial(X_train)
    X_test_poly = pl.polynomial(X_test)
    # Train linear regression model
    theta = np.random.rand(degree).reshape(-1,1)
    model = LinearModel(theta)
    model.fit(X_train_poly, y_train)

    # Save model parameters
    models[degree] = {
        'model': model,
        'degree': degree,
        'mse': model.mse(y_test, model.predict(X_test_poly))
    }

# Evaluate Models on the Test Set
for degree, model_info in models.items():
    print(f'Degree {degree} - MSE: {model_info["mse"]:.4f}')

# Plot Evaluation Curves
degrees, mses = zip(*[(model_info['degree'], model_info['mse']) for model_info in models.values()])
plt.plot(degrees, mses, marker='o')
plt.title('Polynomial Regression Evaluation')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.show()

# Plot True vs Predicted Prices
best_model_degree = min(models, key=lambda k: models[k]['mse'])
best_model = models[best_model_degree]['model']

pl = Polynom(best_model_degree)
X_test_poly_best = pl.polynomial(X_test)
y_pred = best_model.predict(X_test_poly_best)

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.title('True vs Predicted Prices')
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Save Model Parameters into models.csv/yml/pickle
# Implement saving model parameters here
with open('best_model.pkl', 'wb') as f:
    dill.dump(best_model, f)
    
params = {
    'path':'best_model.pkl',
    'theta': best_model.theta.tolist(),
    'degree': models[best_model_degree]['degree'],
} 

with open('best_model_params.yml', 'w') as f:
    yaml.dump(params, f, default_flow_style=False)

