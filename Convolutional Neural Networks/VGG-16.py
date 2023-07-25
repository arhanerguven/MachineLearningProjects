"""
Implementation of the VGG-16 CNN Architecture
by @arhanerguven
"""

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Rescaling
import numpy as np
from keras import layers
from keras.utils import image_dataset_from_directory
import tensorflow as tf

batch_size = 32
img_height = 224
img_width = 224

train_set = image_dataset_from_directory(
 '/content/train',
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_set = image_dataset_from_directory(
  '/content/train',
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_set.class_names
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_set.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_set.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

num_classes = len(class_names)

model = Sequential()
model.add(Rescaling(1./255, input_shape=(img_height, img_width, 3)))
model.add(Conv2D(input_shape=(224 ,224 ,3) ,filters=64 ,kernel_size=(3 ,3) ,padding="same", activation="relu"))
model.add(Conv2D(filters=64 ,kernel_size=(3 ,3) ,padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2 ,2) ,strides=(2 ,2)))
model.add(Conv2D(filters=128, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2 ,2) ,strides=(2 ,2)))
model.add(Conv2D(filters=256, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2 ,2) ,strides=(2 ,2)))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2 ,2) ,strides=(2 ,2)))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3 ,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2 ,2) ,strides=(2 ,2)))
model.add(Flatten())
model.add(Dense(units=4096 ,activation="relu"))
model.add(Dense(units=4096 ,activation="relu"))
model.add(Dense(units=num_classes, activation="softmax"))

model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit_generator(generator=train_set, validation_data= val_set ,epochs=20)

test_set = image_dataset_from_directory('test' ,image_size=(224 ,224))

import tensorflow as tf


predictions = model.predict(test_set)
#enter the length of the test set
test_length = 400
for i in range(test_length):
  score = tf.nn.softmax(predictions[i])
  print(class_names[np.argmax(score)], 10.0 *np.max(score))

#draw accuracy and loss graphs
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

