import tensorflow as tf
from tensorflow import keras

import numpy as np

import glob
import cv2

dog_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/dog/*.jpg")])
cat_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/cat/*.jpg")])
train_dog = dog_files[3000:]
train_cat = cat_files[3000:]
test_dog = dog_files[1000:3000]
test_cat = cat_files[1000:3000]

train_images = np.concatenate((train_dog, train_cat), axis=0)
train_labels = np.concatenate((np.asarray([1 for i in train_dog]), np.asarray([0 for i in train_cat])), axis=0)
test_images = np.concatenate((test_dog, test_cat), axis=0)
test_labels = np.concatenate((np.asarray([1 for i in test_dog]), np.asarray([0 for i in test_cat])), axis=0)

class_names = ['cat', 'dog']

print(keras.backend.image_data_format())

train_images = train_images / 255.0
test_images = test_images / 255.0

input_shape = (64, 64, 3)
num_classes = 2

model = keras.Sequential()

model.add(keras.layers.Conv2D(16, kernel_size=(8, 8),
                              strides=1,
                              activation='relu',
                              input_shape=input_shape))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(32, (5, 5), strides=1, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
l3 = keras.layers.Conv2D(64, (3, 3), strides=1, activation='relu')
model.add(l3)
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
s4 = l3.output_shape[1] * 2
# model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, (s4, s4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
l4 = keras.layers.Conv2D(128, (3, 3), activation='relu')
model.add(l4)
model.add(keras.layers.BatchNormalization())
s4 = l4.output_shape[1] * 2
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, (s4, s4), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
l6 = keras.layers.BatchNormalization()
model.add(l6)

model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

#############################################################
# model.add(keras.layers.concatenate([l3, l6]))
s1 = l6.output_shape[1] * 2
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, (s1, s1), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
l7 = keras.layers.Conv2D(64, (3, 3), activation='relu')
model.add(l7)

print('l6: ')
print(l6.output_shape)
print('\n')
print('l7: ')
print(l7.output_shape)
print('\n')

model.add(keras.layers.BatchNormalization())
s7 = l7.output_shape[1] * 2
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, (s7, s7), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
l8 = keras.layers.Conv2D(32, (3, 3), activation='relu')
model.add(l8)
model.add(keras.layers.BatchNormalization())
s8 = l8.output_shape[1] * 2
model.add(keras.layers.Lambda(lambda x: tf.image.resize_images(x, (s8, s8), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)))
model.add(keras.layers.Conv2D(16, (3, 3), activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.25))

# model.add(keras.layers.Conv2D(128, kernel_size=(3, 3),
#                  activation='relu'))
# model.add(keras.layers.Conv2D(64, (3, 3), activation='sigmoid'))
# model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, shuffle=True)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
