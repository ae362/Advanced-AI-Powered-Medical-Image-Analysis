import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Check available GPUs
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Enable device placement logging
tf.debugging.set_log_device_placement(True)

# Perform a simple TensorFlow operation
a = tf.constant([1.0, 2.0, 3.0], name="a")
b = tf.constant([4.0, 5.0, 6.0], name="b")
c = tf.add(a, b, name="c")
print("Result:", c)
