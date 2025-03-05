import numpy as np  # Import the library for numerical calculations with arrays
import matplotlib.pyplot as plt  # Import the library for plotting
from sklearn.datasets import make_gaussian_quantiles  # Import function to generate classification data

def train_neural_network():
    """
    Main function to train the neural network and visualize the results.
    """
    
    def create_dataset(N=1000):
        """
        Generates a classification dataset with two classes.
        """
        gaussian_quantiles = make_gaussian_quantiles(
            mean=None,  # Mean of the data (automatic)
            cov=0.1,  # Variance of the data
            n_samples=N,  # Number of samples to generate
            n_features=2,  # Number of features per sample
            n_classes=2,  # Number of classes
            shuffle=True,  # Shuffle the data
            random_state=None  # No fixed seed for randomness
        )
        X, Y = gaussian_quantiles  # Separate features and labels
        Y = Y[:, np.newaxis]  # Convert labels to column matrix
        return X, Y

    def sigmoid(x, derivate=False):
        """
        Sigmoid activation function.
        """
        if derivate:
            return np.exp(-x) / (np.exp(-x) + 1)**2  # Derivative of sigmoid
        else:
            return 1 / (1 + np.exp(-x))  # Calculate the sigmoid

    def relu(x, derivate=False):
        """
        ReLU activation function.
        """
        if derivate:
            x[x <= 0] = 0  # The derivative is 0 for negative values
            x[x > 0] = 1  # The derivative is 1 for positive values
            return x
        else:
            return np.maximum(0, x)  # Return the maximum between 0 and x

    def mse(y, y_hat, derivate=False):
        """
        Loss function: Mean Squared Error.
        """
        if derivate:
            return (y_hat - y)  # Derivative of MSE
        else:
            return np.mean((y_hat - y)**2)  # Calculate MSE

    def initialize_parameters_deep(layers_dims):
        """
        Initializes the weights and biases of the neural network.
        """
        parameters = {}  # Dictionary to store parameters
        L = len(layers_dims)  # Number of layers
        for l in range(0, L-1):
            parameters['W' + str(l+1)] = (np.random.rand(layers_dims[l], layers_dims[l+1]) * 2) - 1  # Weights
            parameters['b' + str(l+1)] = (np.random.rand(1, layers_dims[l+1]) * 2) - 1  # Biases
        return parameters

    def train(x_data, y_data, learning_rate, params, training=True):
        """
        Performs forward propagation and, if enabled, the training of the network.
        """
        params['A0'] = x_data  # Input to the neural network

        params['Z1'] = np.matmul(params['A0'], params['W1']) + params['b1']  # Calculate the first layer
        params['A1'] = relu(params['Z1'])  # Apply ReLU

        params['Z2'] = np.matmul(params['A1'], params['W2']) + params['b2']  # Calculate the second layer
        params['A2'] = relu(params['Z2'])  # Apply ReLU

        params['Z3'] = np.matmul(params['A2'], params['W3']) + params['b3']  # Calculate the third layer
        params['A3'] = sigmoid(params['Z3'])  # Apply sigmoid

        output = params['A3']  # Final output of the network

        if training:
            # Calculate the error and backpropagate
            params['dZ3'] = mse(y_data, output, True) * sigmoid(params['A3'], True)
            params['dW3'] = np.matmul(params['A2'].T, params['dZ3'])

            params['dZ2'] = np.matmul(params['dZ3'], params['W3'].T) * relu(params['A2'], True)
            params['dW2'] = np.matmul(params['A1'].T, params['dZ2'])

            params['dZ1'] = np.matmul(params['dZ2'], params['W2'].T) * relu(params['A1'], True)
            params['dW1'] = np.matmul(params['A0'].T, params['dZ1'])

            # Update weights and biases using gradient descent
            params['W3'] -= params['dW3'] * learning_rate
            params['W2'] -= params['dW2'] * learning_rate
            params['W1'] -= params['dW1'] * learning_rate

            params['b3'] -= np.mean(params['dW3'], axis=0, keepdims=True) * learning_rate
            params['b2'] -= np.mean(params['dW2'], axis=0, keepdims=True) * learning_rate
            params['b1'] -= np.mean(params['dW1'], axis=0, keepdims=True) * learning_rate

        return output  # Return output

    # Create the dataset
    X, Y = create_dataset()
    layers_dims = [2, 6, 10, 1]  # Define the architecture of the network
    params = initialize_parameters_deep(layers_dims)  # Initialize parameters
    error = []  # List to store errors

    for _ in range(50000):  # Training the neural network
        output = train(X, Y, 0.001, params)  # Train with learning rate 0.001
        if _ % 50 == 0:
            print(mse(Y, output))  # Print error every 50 iterations
            error.append(mse(Y, output))  # Store error in the list

    # Plot the training data
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Spectral)

    # Create new test data
    data_test_x = (np.random.rand(1000, 2) * 2) - 1
    data_test_y = train(data_test_x, X, 0.0001, params, training=False)

    y = np.where(data_test_y > 0.5, 1, 0)  # Classify test data
    plt.scatter(data_test_x[:, 0], data_test_x[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()  # Show plot

if __name__ == "__main__":
    train_neural_network()  # Run the main function