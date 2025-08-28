import os, datetime
import tensorflow as tf
import keras
from keras import layers, callbacks

print("TF:", tf.__version__)
print("Keras:", keras.__version__)

# ---- tiny dummy dataset ----
x = tf.random.normal([256, 4])
y = tf.reduce_sum(x, axis=1, keepdims=True)

# ---- simple model ----
m = keras.Sequential([
    layers.Dense(8, activation="relu"),
    layers.Dense(1)
])
m.compile(optimizer="adam", loss="mse")

# ---- TensorBoard callback ----
logdir = os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = callbacks.TensorBoard(log_dir=logdir)

# ---- train ----
m.fit(x, y, epochs=5, batch_size=32, callbacks=[tb], verbose=1)

# ---- save ----
m.save("model.keras")

print("Done ✅  Logs:", logdir)