import numpy as np
import sys

class Percepetron:

    def __init__(self, training_inputs: np.ndarray, training_outputs: np.ndarray, lr=0.01) -> None:
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        # Initialize weights for each input feature with small random values
        self.synaptic_weights = 2 * np.random.random((self.training_inputs.shape[1], 1)) - 1
        self.lr = lr
    def sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x) -> float:
        return x * (1 - x)

    def back_propagation_alg(self) -> None:
        # Forward pass: calculate outputs
        outputs = self.sigmoid(np.dot(self.training_inputs, self.synaptic_weights))
        error = self.training_outputs - outputs
        adjustments = self.lr * error * self.sigmoid_derivative(outputs)
        # Update weights with adjustments
        self.synaptic_weights += np.dot(self.training_inputs.T, adjustments)
        return outputs

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        # Forward pass to make predictions for multiple inputs
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return (output > 0.5).astype(int)  # Apply threshold of 0.5 for binary classification

    def accuracy(self, test_inputs: np.ndarray, test_outputs: np.ndarray) -> float:
        predictions = self.predict(test_inputs)
        correct_predictions = np.sum(predictions == test_outputs)
        accuracy = correct_predictions / len(test_outputs) * 100
        
        # Print class distribution of predictions for debugging
        print("Predictions distribution:", np.unique(predictions, return_counts=True))
        print("Actual distribution:", np.unique(test_outputs, return_counts=True))
        
        return accuracy
    def train(self, iterations=10000) -> None:
        print("Initial synaptic weights:")
        print(self.synaptic_weights)
        
        # Training over specified number of iterations
        for i in range(iterations):
            print(f"Iterations: {i}/{iterations}", end="\r")
            sys.stdout.flush()
            outputs = self.back_propagation_alg()

        print("Post-training synaptic weights:")
        print(self.synaptic_weights)
        print("Training outputs after training:")
        print(outputs)

class LoadModel():
    def __init__(self, model_path) -> None:
        self.model_math = model_path

    

"""
import numpy as np

class Perceptron:
    def __init__(self, training_inputs: np.ndarray, training_outputs: np.ndarray, lr=0.1) -> None:
        self.training_inputs = training_inputs
        self.training_outputs = training_outputs
        # Initialize weights for each input feature with small random values
        self.synaptic_weights = 2 * np.random.random((self.training_inputs.shape[1], 1)) - 1
        self.lr = lr

    def sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, output) -> float:
        return output * (1 - output)

    def train(self, epochs=100, batch_size=32) -> None:
        print("Initial synaptic weights:")
        print(self.synaptic_weights)

        for epoch in range(epochs):
            # Shuffle data at the start of each epoch
            indices = np.arange(self.training_inputs.shape[0])
            np.random.shuffle(indices)
            shuffled_inputs = self.training_inputs[indices]
            shuffled_outputs = self.training_outputs[indices]

            # Batch training
            for start_idx in range(0, len(shuffled_inputs), batch_size):
                end_idx = min(start_idx + batch_size, len(shuffled_inputs))
                batch_inputs = shuffled_inputs[start_idx:end_idx]
                batch_outputs = shuffled_outputs[start_idx:end_idx]
                
                # Forward and backward pass for each batch
                outputs = self.sigmoid(np.dot(batch_inputs, self.synaptic_weights))
                error = batch_outputs - outputs
                adjustments = self.lr * error * self.sigmoid_derivative(outputs)
                self.synaptic_weights += np.dot(batch_inputs.T, adjustments)

            # Print accuracy after each epoch for tracking
            epoch_accuracy = self.accuracy(self.training_inputs, self.training_outputs)
            print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {epoch_accuracy:.2f}%")

        print("Post-training synaptic weights:")
        print(self.synaptic_weights)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return (output > 0.5).astype(int)

    def accuracy(self, test_inputs: np.ndarray, test_outputs: np.ndarray) -> float:
        predictions = self.predict(test_inputs)
        correct_predictions = np.sum(predictions == test_outputs)
        accuracy = correct_predictions / len(test_outputs) * 100
        return accuracy
"""