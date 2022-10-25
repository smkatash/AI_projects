import cv2 as cv
import numpy as np
from matplotlib import pyplot
from tensorflow import keras
from keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_lables) = datasets.cifar10.load_data()

'''
tf.keras.datasets.cifar10.load_data()
Returns
Tuple of NumPy arrays: (x_train, y_train), (x_test, y_test).
'''

training_images = training_images / 255
testing_images = testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for img in range(16):
    pyplot.subplot(4, 4, img+1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.imshow(training_images[img], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[training_labels[img][0]])
pyplot.show()