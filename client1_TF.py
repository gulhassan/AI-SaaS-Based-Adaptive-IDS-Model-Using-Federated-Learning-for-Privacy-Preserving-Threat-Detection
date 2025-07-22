import flwr as fl
import numpy as np
import sys
sys.path.append("/home/gul/nsl_kdd")
from nsl_kdd_preprocess import load_nsl_kdd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# 1. Load data
X_train, X_test, y_train, y_test, _, _ = load_nsl_kdd(
    "/home/gul/nsl_kdd/KDDTrain+_balanced.txt",
    "/home/gul/nsl_kdd/KDDTest+_balanced.txt",
    client_id=0,   # change for client2, client3, etc.
    num_clients=3
)

# 2. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build the Keras model
def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )
    return model

model = build_model(X_train.shape[1])

# 4. Flower Keras client
class KerasClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)
        return model.get_weights(), len(X_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
        return float(loss), len(X_test), {"accuracy": float(accuracy)}

# 5. Start Flower client
fl.client.start_numpy_client(server_address="13.60.24.113:8080", client=KerasClient())

