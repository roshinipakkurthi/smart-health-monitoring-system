import tensorflow as tf
from pathlib import Path

saved_model_dir = 'models/tf/anomaly_tf_model'
tflite_out = Path('tinyml/anomaly_model.tflite')
tflite_out.parent.mkdir(exist_ok=True, parents=True)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# Enable optimizations for TinyML
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(tflite_out, 'wb') as f:
    f.write(tflite_model)

print(f'Wrote: {tflite_out}')
