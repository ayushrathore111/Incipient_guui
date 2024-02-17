#grey wolf optimization with knn

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
# Grey Wolf Optimization
def gwo(X_train, y_train, n_wolves, max_iterations, n_neighbors):
    n_features = X_train.shape[1]
    n_samples = X_train.shape[0]

    # Initialize the positions of alpha, beta, and delta wolves randomly
    alpha_position = np.random.rand(n_features)
    beta_position = np.random.rand(n_features)
    delta_position = np.random.rand(n_features)
    print(alpha_position)
    for iteration in range(max_iterations):
        a = 2 - 2 * (iteration / max_iterations)  # a decreases linearly from 2 to 0

        for i in range(n_wolves):
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)

            A1 = 2 * a * r1 - a  # Randomization vector
            C1 = 2 * r2  # Randomization vector

            D_alpha = np.abs(C1 * alpha_position - X_train[i])
            X1 = alpha_position - A1 * D_alpha

            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)

            A2 = 2 * a * r1 - a
            C2 = 2 * r2

            D_beta = np.abs(C2 * beta_position - X_train[i])
            X2 = beta_position - A2 * D_beta

            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)

            A3 = 2 * a * r1 - a
            C3 = 2 * r2

            D_delta = np.abs(C3 * delta_position - X_train[i])
            X3 = delta_position - A3 * D_delta

            # Update the position of the current wolf
            X_new = (X1 + X2 + X3) / 3

            # Evaluate the fitness of the new position
            knn = KNeighborsRegressor(n_neighbors=n_neighbors)
            knn.fit(X_train * X_new, y_train)
            accuracy = r2_score(y_train, knn.predict(X_train * X_new))

            # Update alpha, beta, and delta positions
            if accuracy > r2_score(y_train, knn.predict(X_train * alpha_position)):
                alpha_position = X_new.copy()
            elif accuracy > r2_score(y_train, knn.predict(X_train * beta_position)):
                beta_position = X_new.copy()
            elif accuracy > r2_score(y_train, knn.predict(X_train * delta_position)):
                delta_position = X_new.copy()

    return alpha_position

# Load dataset and split into training and testing sets
import pandas as pd
df = pd.read_excel("Incipient motion.xlsx")
df=df.dropna()
X = df.iloc[:,:-1]

y = df['tb']
from sklearn import preprocessing
from sklearn import utils

#convert y values to categorical values
lab = preprocessing.LabelEncoder()
y_transformed = lab.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.2, random_state=0)

n_wolves = 7
max_iterations = 100
n_neighbors = 3

# Run GWO to optimize KNN
best_solution = gwo(X_train, y_train, n_wolves, max_iterations,3)

# Train the KNN model with the best solution
knn = BaggingRegressor()
knn.fit(X_train * best_solution, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test * best_solution)

# Evaluate the model
accuracy = r2_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
