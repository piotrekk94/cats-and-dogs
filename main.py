import tensorflow as tf
from tensorflow import keras

import numpy as np

import glob
import cv2

dog_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/dog/*.jpg")])
cat_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/cat/*.jpg")])

images_src = np.concatenate((dog_files, cat_files), axis=0)
labels_src = np.concatenate((np.asarray([1 for i in dog_files]), np.asarray([0 for i in cat_files])), axis=0)

images = images_src[3000:]
labels = labels_src[3000:]

test_img = images_src[1000:3000]
test_labels = labels_src[1000:3000]

class_names = ['cat', 'dog']

images = images / 255.0

print(keras.backend.image_data_format())

input_shape = (64, 64, 3)
num_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images, labels, epochs=5, validation_split=0.2)

test_loss, test_acc = model.evaluate(test_img, test_labels)

print('Test accuracy:', test_acc)