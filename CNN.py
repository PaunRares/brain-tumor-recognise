#importuri
import tensorflow as tf
import os
import cv2
import numpy as np
import imutils
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D, Dropout, Dense
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#prelucrare imagini

def crop_contur(image):
      grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      grayscale1 = cv2.GaussianBlur(grayscale, (5,5), 0)
      threshold_image = cv2.threshold(grayscale1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
      threshold_image = cv2.erode(threshold_image,None,iterations=2)
      threshold_image = cv2.dilate(threshold_image,None,iterations=2)
      contur = cv2.findContours(threshold_image.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      contur = imutils.grab_contours(contur)
      c = max(contur, key=cv2.contourArea)
      punct_extrm_stanga = tuple(c[c[:, :, 0].argmin()][0])
      punct_extrm_dreapta = tuple(c[c[:, :, 0].argmax()][0])
      punct_extrm_sus = tuple(c[c[:, :, 1].argmin()][0])
      punct_extrm_jos = tuple(c[c[:, :, 1].argmax()][0])

      new_image = grayscale1[punct_extrm_sus[1]:punct_extrm_jos[1], punct_extrm_stanga[0]:punct_extrm_dreapta[0]]
      new_image = cv2.resize(new_image,(28,28))

      return new_image



#etichetare imagini
os.chdir('C:\\Users\\rares\\PycharmProjects\\LICENTA\\yes')
x = []
y = []
for i in os.listdir():
      img = cv2.imread(i)
      img = crop_contur(img)
      x.append(img)
      y.append((i[0:1]))
os.chdir('C:\\Users\\rares\\PycharmProjects\\LICENTA\\no')
for i in os.listdir():
      img = cv2.imread(i)
      img = crop_contur(img)
      x.append(img)
      y.append(i[0:1])

#impartirea imaginilor in train si test(169 train,84 test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
print ("Forma unei imagini din X_train: ", x_train[0].shape)
print ("Forma unei imagini din X_test: ", x_test[0].shape)

#convertirea etichetelor in categorii si transformarea in array-uri a datasetu-lui
le = preprocessing.LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)
y_train = np.array(y_train)
x_train = np.array(x_train)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = np.array(x_test)
y_test = np.array(y_test)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
print("X_train Shape: ", x_train.shape)
print("X_test Shape: ", x_test.shape)
print("y_train Shape: ", y_train.shape)
print("y_test Shape: ", y_test.shape)

#creare model
model=Sequential()
model.add(BatchNormalization(input_shape = (28,28,1)))
model.add(Convolution2D(32, kernel_size = 3, activation ='relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Convolution2D(filters=64, kernel_size=4, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Convolution2D(filters=128, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=128,activation = 'relu'))
model.add(Dense(units = 64, activation = 'relu'))
model.add(Dense(units = 32, activation = 'relu'))
model.add(Dense(units = 2, activation = 'softmax'))
print(model.summary())

#compilarea modelului
model.compile(optimizer='adam', loss = 'categorical_crossentropy',metrics = ['accuracy'])

#antrenarea modelului
history = model.fit(x_train,y_train,epochs=50,validation_data=(x_test,y_test),verbose = 1,initial_epoch=0)

#evaluarea modelului
model.evaluate(x_test,y_test)
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


#acuratetea modelului
loss,acc=model.evaluate(x_test,y_test)
print(acc*100)

#matricea de confuzie
y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)

print(confusion_matrix(np.argmax(y_test,axis=1), y_pred))

#salvara modelului
os.chdir('C:\\Users\\rares\\PycharmProjects\\Licenta')
model.save("Model.h5")





