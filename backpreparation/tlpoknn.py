#TLPO with knn

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define the fitness function using KNN
def fitness_function(X, y, solution, n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X * solution, y)
    y_pred = knn.predict(X * solution)
    return accuracy_score(y, y_pred)

# Initialize the population
population_size = 20
n_features = 8  # Number of features in the dataset
population = np.random.randint(2, size=(population_size, n_features))

# Load a dataset and split it into training and testing sets
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
X_train, X_test, y_train, y_test = train_test_split(X, y_transformed, test_size=0.3, random_state=42)

# Parameters
max_generations = 1000
n_neighbors = 3  # K for KNN

# Main loop
for generation in range(max_generations):
    # Evaluate fitness for each individual in the population
    fitness_values = [fitness_function(X_train, y_train, individual, n_neighbors) for individual in population]

    # Teaching phase
    best_individual = population[np.argmax(fitness_values)]

    for i in range(population_size):
        if i != np.argmax(fitness_values):
            r = np.random.random()
            population[i] = population[i] + r * (best_individual - population[i])

    # Learning phase
    for i in range(population_size):
        partner = np.random.choice(range(population_size))
        while partner == i:
            partner = np.random.choice(range(population_size))
        r = np.random.random()
        population[i] = population[i] + r * (population[i] - population[partner])

# Select the best individual
best_individual = population[np.argmax(fitness_values)]
best_fitness = fitness_function(X_train, y_train, best_individual, n_neighbors)

print("Best Individual:", best_individual)
print("Best Fitness:", best_fitness)

