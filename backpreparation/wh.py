import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error

# Define the Whale Optimization Algorithm
def whale_optimization_algorithm(obj_func, bounds, num_iter=100, num_whales=5):
    dim = len(bounds)
    whales_position = np.random.rand(num_whales, dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    whales_fitness = np.zeros(num_whales)

    for iteration in range(num_iter):
        a = 2 - 2 * iteration / num_iter  # a decreases linearly from 2 to 0

        for i in range(num_whales):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A = 2 * a * r1 - a  # Equation (2.3)
            C = 2 * r2  # Equation (2.4)

            # Update position
            D = np.abs(C * whales_position[i, :] - whales_position[i, :])
            new_position = whales_position[i, :] - A * D  # Equation (2.5)

            # Ensure the new position is within the bounds
            new_position = np.maximum(np.minimum(new_position, bounds[:, 1]), bounds[:, 0])

            # Evaluate the fitness of the new position
            new_fitness = obj_func(new_position)

            # Update the position and fitness if the new position is better
            if new_fitness < whales_fitness[i]:
                whales_position[i, :] = new_position
                whales_fitness[i] = new_fitness

    # Find the best whale
    best_whale_index = np.argmin(whales_fitness)
    best_whale_position = whales_position[best_whale_index, :]

    return best_whale_position

# Example objective function (you should replace this with your regression task)
def objective_function(x):
    # Assuming x is the hyperparameter vector for your regression model
    # For demonstration purposes, we use a simple function
    return np.sum(x**2)

# Example usage with Bagging Regression
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_excel('Incipient motion.xlsx')
    df = df.dropna()
    X=df.iloc[:,:-1]
    y= df.iloc[:,-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the bagging regression model (replace this with your actual regression model)
    base_regressor = BaggingRegressor(n_estimators=10, random_state=42)

    # Define the objective function for hyperparameter tuning
    def objective_function(hyperparameters):
        base_regressor.set_params(**dict(zip(['base_estimator__' + str(i) for i in range(len(hyperparameters))], hyperparameters)))
        base_regressor.fit(X_train, y_train)
        predictions = base_regressor.predict(X_test)
        return mean_squared_error(y_test, predictions)

    # Run the Whale Optimization Algorithm
    best_hyperparameters = whale_optimization_algorithm(objective_function, df)

    print("Best Hyperparameters:", best_hyperparameters)



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import r2_score

# Whale Optimization Algorithm
def whale_optimization_algorithm(obj_func, bounds, num_iter=100, num_whales=5):
    # ... (same as before)
    dim = len(bounds)
    whales_position = np.random.rand(num_whales, dim) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
    whales_fitness = np.zeros(num_whales)

    for iteration in range(num_iter):
        a = 2 - 2 * iteration / num_iter  # a decreases linearly from 2 to 0

        for i in range(num_whales):
            r1 = np.random.random(dim)
            r2 = np.random.random(dim)
            A = 2 * a * r1 - a  # Equation (2.3)
            C = 2 * r2  # Equation (2.4)

            # Update position
            D = np.abs(C * whales_position[i, :] - whales_position[i, :])
            new_position = whales_position[i, :] - A * D  # Equation (2.5)

            # Ensure the new position is within the bounds
            new_position = np.maximum(np.minimum(new_position, bounds[:, 1]), bounds[:, 0])

            # Evaluate the fitness of the new position
            new_fitness = obj_func(new_position)

            # Update the position and fitness if the new position is better
            if new_fitness < whales_fitness[i]:
                whales_position[i, :] = new_position
                whales_fitness[i] = new_fitness

    # Find the best whale
    best_whale_index = np.argmin(whales_fitness)
    best_whale_position = whales_position[best_whale_index, :]

    return best_whale_position

# Objective function for R2 maximization
def objective_function(hyperparameters):
    # Set hyperparameters for BaggingRegressor
    base_regressor.set_params(**dict(zip(['base_estimator__' + str(i) for i in range(len(hyperparameters))], hyperparameters)))
    
    # Fit the model
    base_regressor.fit(X_train, y_train)
    
    # Make predictions on the test set
    predictions = base_regressor.predict(X_test)
    
    # Calculate R2 score
    r2 = r2_score(y_test, predictions)
    
    # Return negative R2 since we want to maximize it
    return -r2

# Example usage with Bagging Regression and Excel data
if __name__ == "__main__":
    # Read data from Excel file
    excel_file_path = 'Incipient motion.xlsx'
    df = pd.read_excel(excel_file_path)

    # Assume the last column is the output variable and the rest are input variables
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the bagging regression model (replace this with your actual regression model)
    base_regressor = BaggingRegressor(n_estimators=10, random_state=42)

    # ... (same as before)

    # Run the Whale Optimization Algorithm
    best_hyperparameters = whale_optimization_algorithm(objective_function, hyperparameter_bounds)

    # Print the best hyperparameters and the corresponding R2 value
    print("Best Hyperparameters:", best_hyperparameters)
    
    # Set the best hyperparameters for the final model
    base_regressor.set_params(**dict(zip(['base_estimator__' + str(i) for i in range(len(best_hyperparameters))], best_hyperparameters)))

    # Fit the final model
    base_regressor.fit(X_train, y_train)

    # Make predictions on the test set using the final model
    final_predictions = base_regressor.predict(X_test)

    # Calculate and print the R2 value
    final_r2 = r2_score(y_test, final_predictions)
    print("Best R2 Value:", final_r2)

