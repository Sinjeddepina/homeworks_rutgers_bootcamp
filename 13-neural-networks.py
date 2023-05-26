import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Create a list of categorical variables 
categorical_variables = # YOUR CODE HERE

# Display the categorical variables list
# YOUR CODE HERE
print(categorical_variables)

# Create a OneHotEncoder instance
enc = OneHotEncoder()

# Encode the categorical variables using OneHotEncoder
encoded_data = enc.fit_transform(categorical_variables)

# Create a DataFrame with the encoded variables
encoded_df = pd.DataFrame(encoded_data.toarray())

# Review the DataFrame
# YOUR CODE HERE
encoded_df.head()

# Add the numerical variables from the original DataFrame to the one-hot encoding DataFrame
encoded_df = pd.concat([encoded_df, applicant_data_df.select_dtypes(include=['float64', 'int64'])], axis=1)

# Review the DataFrame
encoded_df.head()

# Define the target set y using the IS_SUCCESSFUL column
y = encoded_df['IS_SUCCESSFUL']

# Display a sample of y
y.head()

# Define features set X by selecting all columns but IS_SUCCESSFUL
X = encoded_df.drop('IS_SUCCESSFUL', axis=1)

# Review the features DataFrame
X.head()

# Split the preprocessed data into a training and testing dataset
# Assign the function a random_state equal to 1
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create a StandardScaler instance
scaler = StandardScaler()

# Fit the scaler to the features training dataset
X_train_scaled = scaler.fit_transform(X_train)

# Fit the scaler to the features testing dataset
X_test_scaled = scaler.transform(X_test)

# Define the number of inputs (features) to the model
number_input_features = len(X_train_scaled[0])

# Review the number of features
number_input_features

# Define the number of neurons in the output layer
number_output_neurons = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1 = 80

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2 = 30

# Review the number of hidden nodes in the second layer
hidden_nodes_layer2

# Create the Sequential model instance
nn = Sequential()

# Add the first hidden layer
nn.add(Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation='relu'))

# Add the second hidden layer
nn.add(Dense(units=hidden_nodes_layer2, activation='relu'))

# Add the output layer to the model specifying the number of output neurons and activation function
nn.add(Dense(units=number_output_neurons, activation='sigmoid'))

# Display the Sequential model summary
nn.summary()

# Compile the Sequential model
nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using 50 epochs and the training data
model = nn.fit(X_train_scaled, y_train, epochs=50)


# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test)

# Print the model loss and accuracy
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")
Step 4: Save and export your model to an HDF5 file, and name the file AlphabetSoup.h5.

# Set the model's file path
file_path = "AlphabetSoup.h5"

# Export your model to a HDF5 file
nn.save(file_path)

# Download your model to your computer
files.download(file_path)
Optimize the neural network model
Step 1: Define at least three new deep neural network models (resulting in the original plus 3 optimization attempts). With each, try to improve on your first model’s predictive accuracy.

Rewind Recall that perfect accuracy has a value of 1, so accuracy improves as its value moves closer to 1. To optimize your model for a predictive accuracy as close to 1 as possible, you can use any or all of the following techniques:

Adjust the input data by dropping different features columns to ensure that no variables or outliers confuse the model.

Add more neurons (nodes) to a hidden layer.

Add more hidden layers.

Use different activation functions for the hidden layers.

Add to or reduce the number of epochs in the training regimen.

Alternative Model 1

# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features

# Define the number of neurons in the output layer
number_output_neurons_A1 = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A1 = 64

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A1

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2_A1 = 32

# Review the number of hidden nodes in the second layer
hidden_nodes_layer2_A1

# Define the number of hidden nodes for the third hidden layer
hidden_nodes_layer3_A1 = 16

# Review the number of hidden nodes in the third layer
hidden_nodes_layer3_A1

# Create the Sequential model instance
nn_A1 = Sequential()

# First hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer1_A1, input_dim=number_input_features, activation='relu'))

# Output layer
nn_A1.add(Dense(units=number_output_neurons_A1, activation='sigmoid'))

# Check the structure of the model
nn_A1.summary()

# Compile the Sequential model
nn_A1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using 50 epochs and the training data
fit_model_A1 = nn_A1.fit(X_train_scaled, y_train, epochs=50)

# Define the the number of inputs (features) to the model
number_input_features = len(X_train.iloc[0])

# Review the number of features
number_input_features

# Define the number of neurons in the output layer
number_output_neurons_A1 = 1

# Define the number of hidden nodes for the first hidden layer
hidden_nodes_layer1_A1 = 64

# Review the number of hidden nodes in the first layer
hidden_nodes_layer1_A1

# Define the number of hidden nodes for the second hidden layer
hidden_nodes_layer2_A1 = 32

# Review the number of hidden nodes in the second layer
hidden_nodes_layer2_A1

# Define the number of hidden nodes for the third hidden layer
hidden_nodes_layer3_A1 = 16

# Review the number of hidden nodes in the third layer
hidden_nodes_layer3_A1

# Create the Sequential model instance
nn_A1 = Sequential()

# First hidden layer
nn_A1.add(Dense(units=hidden_nodes_layer1_A1, input_dim=number_input_features, activation='relu'))

# Output layer
nn_A1.add(Dense(units=number_output_neurons_A1, activation='sigmoid'))

# Check the structure of the model
nn_A1.summary()

# Compile the Sequential model
nn_A1.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model using 50 epochs and the training data
fit_model_A1 = nn_A1.fit(X_train_scaled, y_train, epochs=50)
Step 2: After finishing your models, display the accuracy scores achieved by each model, and compare the results.

print("Original Model Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn.evaluate(X_test_scaled, y_test)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

print("Alternative Model 1 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = nn_A1.evaluate(X_test_scaled, y_test)

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

print("Alternative Model 2 Results")

# Evaluate the model loss and accuracy metrics using the evaluate method and the test data
model_loss, model_accuracy = # YOUR CODE HERE

# Display the model loss and accuracy results
print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

 yeayea