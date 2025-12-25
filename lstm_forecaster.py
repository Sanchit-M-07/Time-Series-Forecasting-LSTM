"""
This module implements a SimpleLSTMCell and a TimeSeriesLSTM model for
univariate time series forecasting using NumPy. It includes methods for
single-step forward passes, rolling over input sequences, and multi-step
forecasting by iteratively feeding back predictions as new inputs.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SimpleLSTMCell:
    """Simple LSTM Cell Implementation"""
    
    def __init__(self, input_size, hidden_size):
        self.hidden_size = hidden_size
        
        # Initialize weights
        self.Wxf = np.random.randn(input_size, hidden_size) * 0.01
        self.Whf = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bf = np.zeros((1, hidden_size))
        
        self.Wxi = np.random.randn(input_size, hidden_size) * 0.01
        self.Whi = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bi = np.zeros((1, hidden_size))
        
        self.Wxc = np.random.randn(input_size, hidden_size) * 0.01
        self.Whc = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bc = np.zeros((1, hidden_size))
        
        self.Wxo = np.random.randn(input_size, hidden_size) * 0.01
        self.Who = np.random.randn(hidden_size, hidden_size) * 0.01
        self.bo = np.zeros((1, hidden_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def tanh(self, x):
        return np.tanh(x)
    
    def forward(self, x, h_prev, c_prev):
        """Forward pass"""
        # Forget gate
        f = self.sigmoid(np.dot(x, self.Wxf) + np.dot(h_prev, self.Whf) + self.bf)
        
        # Input gate
        i = self.sigmoid(np.dot(x, self.Wxi) + np.dot(h_prev, self.Whi) + self.bi)
        
        # Candidate cell state
        c_tilde = self.tanh(np.dot(x, self.Wxc) + np.dot(h_prev, self.Whc) + self.bc)
        
        # Cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = self.sigmoid(np.dot(x, self.Wxo) + np.dot(h_prev, self.Who) + self.bo)
        
        # Hidden state
        h = o * self.tanh(c)
        
        return h, c

class TimeSeriesLSTM:
    """LSTM for Time Series Forecasting"""
    
    def __init__(self, sequence_length=10, n_features=1, n_hidden=50, learning_rate=0.01):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        
        self.lstm = SimpleLSTMCell(n_features, n_hidden)
        
        # Output layer
        self.W_output = np.random.randn(n_hidden, 1) * 0.01
        self.b_output = np.zeros((1, 1))
    
    def create_sequences(self, data):
        """Create sequences for training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i+self.sequence_length])
            y.append(data[i+self.sequence_length])
        return np.array(X), np.array(y)
    
    def forward(self, X):
        """Forward pass"""
        h, c = np.zeros((1, self.n_hidden)), np.zeros((1, self.n_hidden))
        
        for t in range(self.sequence_length):
            x_t = X[:, t:t+1, :]
            h, c = self.lstm.forward(x_t, h, c)
        
        output = np.dot(h, self.W_output) + self.b_output
        return output, h
    
    def fit(self, X_train, y_train, epochs=50):
        """Train the model"""
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X_train)):
                # Forward pass
                pred, h = self.forward(X_train[i:i+1])
                
                # Loss
                loss = (pred - y_train[i:i+1]) ** 2
                total_loss += loss
                
                # Simple gradient update
                self.W_output -= self.learning_rate * 2 * (pred - y_train[i:i+1]) * h.T
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.mean():.6f}")
    
    def predict(self, X):
        """Make predictions"""
        predictions = []
        for i in range(len(X)):
            pred, _ = self.forward(X[i:i+1])
            predictions.append(pred[0, 0])
        return np.array(predictions)

# Usage Example
if __name__ == "__main__":
    # Generate synthetic time series data
    np.random.seed(42)
    t = np.arange(200)
    data = 50 + 10 * np.sin(t/10) + np.random.normal(0, 1, 200)
    
    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    # Create sequences
    model = TimeSeriesLSTM(sequence_length=10, n_
