import numpy as np
from sklearn.ensemble import AdaBoostRegressor,BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

# Objective function for optimization (replace with your own)
def objective_function(params, X_train, y_train, X_val, y_val):
    # Train AdaBoost model
    base_classifier = DecisionTreeRegressor(max_depth=1)
    adaboost_model = BaggingRegressor(base_classifier, n_estimators=100)
    adaboost_model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = adaboost_model.predict(X_val)
    mse= mean_squared_error(y_pred,y_val)
    return mse

# Harris Cultural Algorithm
def harris_cultural_algorithm_with_adaboost(X_train, y_train, X_val, y_val, population_size, num_iterations):
    num_dimensions = 2  # Number of hyperparameters for AdaBoost: (n_estimators, learning_rate)
    lower_bound = [10, 0.01]  # Lower bounds for hyperparameters
    upper_bound = [200, 1.0]  # Upper bounds for hyperparameters

    # Initialize population within bounds
    population = np.random.uniform(low=lower_bound, high=upper_bound, size=(population_size, num_dimensions))

    for iteration in range(num_iterations):
        # Evaluate objective function for each individual
        fitness = np.apply_along_axis(objective_function, 1, population, X_train, y_train, X_val, y_val)

        # Sort individuals based on fitness (maximization task)
        sorted_indices = np.argsort(fitness)
        population = population[sorted_indices]

        # Cultural Knowledge Transfer (e.g., crossover, mutation)
        for i in range(1, population_size):
            alpha = np.random.rand()
            population[i] = alpha * population[i] + (1 - alpha) * population[i - 1]

        # Exploration (e.g., random perturbation)
        exploration_rate = 0.1
        exploration = exploration_rate * np.random.uniform(low=-1, high=1, size=(population_size, num_dimensions))
        population += exploration

    # Return the best individual and its fitness value
    best_hyperparameters = population[0]
    best_fitness = -fitness[0]  # Undo the negation for accuracy maximization

    return best_hyperparameters, best_fitness

if __name__ == "__main__":
    # Generate synthetic data for demonstration
    # Load the dataset from an Excel file
    import pandas as pd
    df = pd.read_excel('Incipient motion.xlsx')
    df = df.dropna()
    X=df.iloc[:,:-1]
    y= df.iloc[:,-1]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set hyperparameters
    population_size = 50
    num_iterations = 100

    # Run Harris Cultural Algorithm with AdaBoost
    best_hyperparameters, best_fitness = harris_cultural_algorithm_with_adaboost(X_train, y_train, X_val, y_val, population_size, num_iterations)

    print("Best Hyperparameters:", best_hyperparameters)
    print("Best Fitness (Accuracy):", best_fitness)
