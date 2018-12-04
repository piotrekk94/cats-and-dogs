# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

#load images
import glob
import cv2

fashion_mnist = keras.datasets.fashion_mnist

dog_files = np.asarray([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in glob.glob("dataset/dog/*.jpg")])
cat_files = np.asarray([cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in glob.glob("dataset/cat/*.jpg")])
train_dog = dog_files[3000:]
train_cat = cat_files[3000:]
test_dog = dog_files[1000:3000]
test_cat = cat_files[1000:3000]

train_images = np.concatenate((train_dog, train_cat), axis=0)
train_labels = np.concatenate((np.asarray([1 for i in train_dog]), np.asarray([0 for i in train_cat])), axis=0)
test_images = np.concatenate((test_dog, test_cat), axis=0)
test_labels = np.concatenate((np.asarray([1 for i in test_dog]), np.asarray([0 for i in test_cat])), axis=0)

class_names = ['cat', 'dog']

train_images = train_images / 255.0

test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
#plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(64, 64)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)
