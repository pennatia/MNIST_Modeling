import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

train_data = pd.read_csv('Data/train.csv')

# Prep the Data
mnist_data_train = train_data.iloc[0:30000] / 255
x_train = np.array(mnist_data_train.drop(['label'], axis =1))
x_train = x_train.reshape([len(x_train),28,28])
plt.imshow(x_train[0])
y_train = np.array(mnist_data_train['label'])

mnist_data_test = train_data.iloc[30000:42000] / 255
x_test = np.array(mnist_data_test.drop(['label'], axis =1))
x_test = x_test.reshape([len(x_test),28,28])
y_test = np.array(mnist_data_test['label'])

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
input_shape = (28,28,1)
num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    keras.Input(shape=input_shape), 
    keras.layers.Conv2D(32,kernel_size = (3,3), activation = "relu"),
    keras.layers.MaxPooling2D(pool_size = (2,2)),
    keras.layers.Conv2D(16, kernel_size=(3,3), activation = 'relu' ),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation = 'softmax')])

model.summary()

batch_size = 150
epochs = 15


model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, epochs = epochs, batch_size=batch_size, validation_split=0.01)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
