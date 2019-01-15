import tensorflow as tf
from tensorflow import keras

import numpy as np

from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

    ## As loss always exists
    epochs = range(1,len(history.history[loss_list[0]]) + 1)

    ## Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'r', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.jpg')

    ## Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'r', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy.jpg')
    plt.show()


dog_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/dog/*.jpg")])
cat_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/cat/*.jpg")])

train_dog = dog_files[4000:]
train_cat = cat_files[4000:]
test_dog = dog_files[1000:3000]
test_cat = cat_files[1000:3000]

train_images = np.concatenate((train_dog, train_cat), axis=0)
train_labels = np.concatenate((np.asarray([1 for i in train_dog]), np.asarray([0 for i in train_cat])), axis=0)
test_images = np.concatenate((test_dog, test_cat), axis=0)
test_labels = np.concatenate((np.asarray([1 for i in test_dog]), np.asarray([0 for i in test_cat])), axis=0)

print(keras.backend.image_data_format())

train_images = train_images / 255.0
test_images = test_images / 255.0

input_shape = (64, 64, 3)
num_classes = 2

model = keras.Sequential()

kernel_size = 2
pool_size = 2

model.add(keras.layers.Conv2D(50, kernel_size=kernel_size,
                              strides=1,
                              activation='relu',
                              input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

model.add(keras.layers.Conv2D(70, kernel_size=kernel_size,
                              strides=1,
                              activation='relu',
                              input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

model.add(keras.layers.Conv2D(90, kernel_size=kernel_size,
                              strides=1,
                              activation='relu',
                              input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

model.add(keras.layers.Conv2D(120, kernel_size=kernel_size,
                              strides=1,
                              activation='relu',
                              input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=pool_size))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='sigmoid'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=20, shuffle=True)

model.summary()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Check if the current model has better performance than the highest according to accuracy.dat
fs = open('accuracy.dat', 'r')
fs_acc = fs.readline()
print('Actual maximum accuracy: ' + fs_acc)
fs.close()

if float(fs_acc) < float(test_acc):
    # serialize model to JSON
    model_json = model.to_json()
    with open(str(int(test_acc*100)) + "_model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(str(int(test_acc*100)) + "_model.h5")
    print("Saved model to disk")
    new_acc = open('accuracy.dat', 'w')
    new_acc.write(str(test_acc))
    new_acc.close()

# Some plots from training
plot_history(history)
