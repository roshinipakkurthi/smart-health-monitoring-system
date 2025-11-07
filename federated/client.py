import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('--client_id', type=int, default=1)
args = parser.parse_args()

# Load and split a different slice per client (simulate data silo)
data = pd.read_csv('data/health_data.csv')
X = data.drop('anomaly', axis=1).values.astype('float32')
y = data['anomaly'].values.astype('float32')

# Simple shard by client_id
shard_size = len(X) // 2 if len(X) >= 2 else len(X)
start = (args.client_id - 1) * shard_size
end = start + shard_size
X_client = X[start:end]
y_client = y[start:end]

X_train, X_test, y_train, y_test = train_test_split(X_client, y_client, test_size=0.2, random_state=42)

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class Client(fl.client.NumPyClient):
    def __init__(self):
        self.model = get_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)
        return self.model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(X_test, y_test, verbose=0)
        return float(loss), len(X_test), {"accuracy": float(acc)}

if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=Client())
