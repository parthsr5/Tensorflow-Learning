import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train.shape)
# print(y_train.shape)

x_train = x_train.reshape(-1, 784).astype("float32")  / 255.0 # flatten the 60000*28*28 to 60000*784
    # -1 means keep that particular dimension unchanged
    # convert tto float 32
    # normalise to get values between 0 and 1
x_test = x_test.reshape(-1, 784).astype("float32") / 255.

# x_train = tf.convert_to_tensor(x_train)
# optional to convert, as tf will convert to tensor automatically
# originally data loaded is in numpy array




# Sequential API (very convenient, not very flexible)
    # 1 input can be mapped to 1 output, major limitation

model = keras.Sequential(
    [
        keras.Input(shape=(28*28)),
            # putting this here allows us to view model.summary, otherwise it can be done post model.fit
        layers.Dense(512, activation = 'relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]
)

# print(model.summary())
#
# import sys
# sys.exit()

# another way to do it is to add one layer at a time
# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layer.Dense(10))
# print(model.summary()) # this is a good debugging tool


model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs = 5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


    # Extracting specific layer features
# model = keras.Model(inputs=model.inputs, outputs=[model.layers[-2].output]) # another way is outputs=[model.get_layer('name_of_layer').output]
#                                                         # to get all layers, outputs = [layer.ouput for layer in model.layers]
# feature = model.predict(x_train)
# print(feature.shape)
#
#
# import sys
# sys.exit()

# ===============================================




# Functional API ( A bit more flexible)
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation = 'relu', name='first_layer')(inputs)
x = layers.Dense(256, activation = 'relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(),
    optimizer = keras.optimizers.Adam(lr=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs = 5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


