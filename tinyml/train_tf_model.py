import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

data = pd.read_csv('data/health_data.csv')
X = data.drop('anomaly', axis=1).values.astype('float32')
y = data['anomaly'].values.astype('float32')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Test Accuracy: {acc:.4f}')

Path('models/tf').mkdir(parents=True, exist_ok=True)
model.save('models/tf/anomaly_tf_model')
print('Saved TF model to models/tf/anomaly_tf_model')
