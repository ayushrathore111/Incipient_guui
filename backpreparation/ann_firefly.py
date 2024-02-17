import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

# Load data from Excel file
excel_file_path = 'Incipient motion.xlsx'
df = pd.read_excel(excel_file_path)

# # Assume the target variable is in the 'target_column' column
# X = df.drop('target_column', axis=1)
# y = df['target_column']
X= df.iloc[:,:-1]
y = df.iloc[:,-1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network model
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to calculate the accuracy based on the weights provided
def calculate_accuracy(weights, X, y):
    model.set_weights(weights)
    predictions = model.predict(X)
    accuracy = np.mean((predictions.squeeze() > 0.5) == y)
    return accuracy


# ... (Previous code)

# Firefly Algorithm
def initialize_fireflies(num_fireflies, num_dimensions):
    return np.random.rand(num_fireflies, num_dimensions)

def move_firefly(current_position, other_position, alpha=1.0, beta=1.0, gamma=0.1):
    r = np.linalg.norm(current_position - other_position)
    beta *= np.exp(-gamma * r**2)
    return current_position + alpha * (beta * (other_position - current_position)) + 0.01 * np.random.randn(*current_position.shape)

def firefly_algorithm(num_fireflies, num_dimensions, max_iterations, X_train, y_train):
    fireflies = initialize_fireflies(num_fireflies, num_dimensions)
    best_weights = None
    best_accuracy = 0.0

    for iteration in range(max_iterations):
        for i in range(num_fireflies):
            for j in range(num_fireflies):
                accuracy_i = calculate_accuracy(fireflies[i], X_train, y_train)
                accuracy_j = calculate_accuracy(fireflies[j], X_train, y_train)

                if accuracy_j > accuracy_i:
                    fireflies[i] = move_firefly(fireflies[i], fireflies[j])

        # Update the best weights if a better solution is found
        current_best_accuracy = calculate_accuracy(fireflies[np.argmax([calculate_accuracy(x, X_train, y_train) for x in fireflies])], X_train, y_train)
        if current_best_accuracy > best_accuracy:
            best_weights = fireflies[np.argmax([calculate_accuracy(x, X_train, y_train) for x in fireflies])]
            best_accuracy = current_best_accuracy

    return best_weights

# Example usage
num_fireflies = 20
num_dimensions = X_train_scaled.shape[1]
max_iterations = 50

best_weights = firefly_algorithm(num_fireflies, num_dimensions, max_iterations, X_train_scaled, y_train)

# Set the best weights to the model
model.set_weights(best_weights)

# Evaluate the model on the test set
test_accuracy = model.evaluate(X_test_scaled, y_test)[1]
print(f"Accuracy on the test set: {test_accuracy}")
