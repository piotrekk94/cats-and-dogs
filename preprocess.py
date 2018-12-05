import numpy as np
import pathlib
import glob
import cv2
import os

imsize = 64

def preproc(fname, folder, grayscale):
	image = cv2.imread(fname)
	if image is None:
		print("Failed to open %s" % fname)
		return
	resized = cv2.resize(image, (imsize, imsize))
	if grayscale:
		gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	else:
		gray = resized
	cv2.imwrite("./dataset/%s/%s" % (folder ,os.path.basename(fname)) ,gray)

pathlib.Path("./dataset/cat").mkdir(parents=True, exist_ok=True)
pathlib.Path("./dataset/dog").mkdir(parents=True, exist_ok=True)

print("Cats")

for fname in glob.glob("./PetImages/Cat/*.jpg"):
	preproc(fname, "cat", False)

print("Dogs")

for fname in glob.glob("./PetImages/Dog/*.jpg"):
	preproc(fname, "dog", False)

