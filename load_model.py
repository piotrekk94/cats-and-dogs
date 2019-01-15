import tensorflow as tf
from tensorflow import keras

import numpy as np

import sys
from PIL import Image
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

dog_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/dog/*.jpg")])
cat_files = np.asarray([cv2.imread(i) for i in glob.glob("dataset/cat/*.jpg")])

test_dog = dog_files[1000:3000]
test_cat = cat_files[1000:3000]

test_images = np.concatenate((test_dog, test_cat), axis=0)
test_labels = np.concatenate((np.asarray([1 for i in test_dog]), np.asarray([0 for i in test_cat])), axis=0)

to_predict = np.concatenate((dog_files[0:50], cat_files[0:50]), axis=0)

# 1st arg of this script is a path to json model
json_arg = str(sys.argv[1])

# load json and create model
json_file = open(json_arg + '.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(json_arg + ".h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

test_loss, test_acc = loaded_model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

# Prediction tests
#preds = model.predict(to_predict)
#print('Predictions: ')
#for (i, j) in zip(preds, to_predict):
#    if i[0] > i[1]:
#        cv2.imshow('CAT', j)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
#    else:
#        cv2.imshow('DOG', j)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
