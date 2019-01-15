import tensorflow as tf
from tensorflow import keras

import numpy as np

import sys
import itertools
import glob
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import model_from_json

def generate_grid(imgs):
    w = 8
    h = 8
    n = w*h
    margin = 20
    img_h, img_w, img_c = imgs[0][0].shape

    #Define the margins in x and y directions
    m_x = margin
    m_y = margin

    #Size of the full size image
    mat_x = img_w * w + m_x * (w - 1)
    mat_y = img_h * h + m_y * (h - 1)

    #Create a matrix of zeros of the right size and fill with 255 (so margins end up white)
    imgmatrix = np.zeros((mat_y, mat_x, img_c),np.uint8)
    imgmatrix.fill(255)

    #Prepare an iterable with the right dimensions
    positions = itertools.product(range(h), range(w))

    for (y_i, x_i), img in zip(positions, imgs):
        x = x_i * (img_w + m_x)
        y = y_i * (img_h + m_y)
        imgmatrix[y:y+img_h, x:x+img_w, :] = img[0]

    resized = cv2.resize(imgmatrix, (mat_x//3,mat_y//3), interpolation = cv2.INTER_AREA)
    compression_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    cv2.imwrite('grid.jpg', imgmatrix, compression_params)


# Datasets
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
preds = loaded_model.predict(to_predict)

processed_list = list()
font = cv2.FONT_HERSHEY_SIMPLEX
for (i, j) in zip(preds, to_predict):
    if i[0] > i[1]:
        cv2.putText(j,'CAT',(2,15), font, 0.5,(0,0,0), 2)
        data = (j, 'CAT')
    else:
        cv2.putText(j,'DOG',(2,15), font, 0.5,(0,0,0), 2)
        data = (j, 'DOG')
    processed_list.append(data)

generate_grid(processed_list)
