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

'''
for img in range(16):
    pyplot.subplot(4, 4, img+1)
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.imshow(training_images[img], cmap=pyplot.cm.binary)
    pyplot.xlabel(class_names[training_labels[img][0]])
pyplot.show()
'''

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_lables = testing_lables[:4000]

'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_lables))

loss, accuracy = model.evaluate(testing_images, testing_lables)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
model.save('image_classifier_model')
'''

model = models.load_model('image_classifier_model')

image = cv.imread('cat.jpg')
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
pyplot.imshow(image, cmap=pyplot.cm.binary)

prediction = model.predict(np.array([image]) / 255)
index = np.argmax(prediction)
print(f"Prediction is {class_names[index]}")
