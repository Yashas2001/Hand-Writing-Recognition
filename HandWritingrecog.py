import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.datasets import mnist
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape = (28,28,1), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss= tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

#Early Stopping
es = EarlyStopping(monitor='val_acc', min_delta=0.01, patience= 4, verbose= 1)

#Model checkpoint
mc = ModelCheckpoint('./bestmodel.h5', monitor= 'val_acc', verbose= 1, save_best_only= True)

cb = [es, mc]

#Training the model

his = model.fit(x_train, y_train, epochs= 50, validation_split= 0.3)
#his = model.fit(x_train, y_train, epochs= 5, validation_split= 0.3, callbacks= cb)
model.save('bestmodel.h5')

saved_model = keras.models.load_model('bestmodel.h5')

score = saved_model.evaluate(x_test, y_test)
print(f'Accuracy of the model is: {score[1]}')