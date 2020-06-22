# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:24:04 2020

@author: subham
"""
import time
import pickle
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
from  pop_up        import index_name
from keras.datasets import mnist
from keras.layers   import Dense
from keras.layers   import Dropout
from keras.layers   import Flatten
from keras.layers   import LeakyReLU
from keras          import models
from keras.layers   import MaxPool2D
from keras.optimizers           import Adam
from keras.layers.convolutional import Conv2D
from notification_sound         import sound


# load (downloaded if needed) the MNIST dataset
#(X_train, y_train), (X_test, y_test) = mnist.load_data()

#OR
# load the downloaded dataset
dataset_train  =pd.read_csv('mnist_train.csv')
dataset_test   =pd.read_csv('mnist_test.csv')

#displaying the digit count for both train and test data
number_values_train  =dataset_train.iloc[:,:1].groupby('label')['label'].count()
number_values_test   =dataset_test.iloc[:,:1].groupby('label')['label'].count()
print(('The amount of digits in training set is: \n {} \n The amount of digits in test set is: \n {} ').
      format(number_values_train,number_values_test))

#plotting the digit count for both train and test data
fig, ax = plt.subplots(figsize=(8,5))
p1 = ax.bar(number_values_train.index,number_values_train.values)
p2 = ax.bar(number_values_test.index,number_values_test.values)
l = ax.legend([p1,p2],['Train data', 'Test data'])
plt.xlabel("Digits available")
plt.ylabel("Frequency of the Digit")
plt.title("Digits and their count") 

#splitting the training data
dataset_train_X  =np.asarray(dataset_train.iloc[:,1:]).reshape([len(dataset_train), 28, 28, 1])
dataset_train_Y  =np.asarray(dataset_train.iloc[:,:1]).reshape([len(dataset_train), 1])

#splitting the test data
dataset_test_X  =np.asarray(dataset_test.iloc[:,1:]).reshape([len(dataset_test), 28, 28, 1])
dataset_test_Y  =np.asarray(dataset_test.iloc[:,:1]).reshape([len(dataset_test), 1])

#converting pixel value in the range 0 to 1
dataset_train_X  =dataset_train_X/255
dataset_test_X   =dataset_test_X/255


#visualizing some of the digits
index=index_name()
plt.imshow(dataset_train_X[index].reshape([28,28]),cmap="Blues")
plt.title(('The number is:',str(dataset_train_Y[index])), y=-0.15,color="green")


model = models.Sequential()
print('Waiting for 2 minute')
time.sleep(120)

# Block 1
model.add(Conv2D(32,3, padding  ="same",input_shape=(28,28,1)))
model.add(LeakyReLU())

model.add(Conv2D(32,3, padding  ="same"))
model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())

model.add(Conv2D(64,3, padding  ="same"))
model.add(LeakyReLU())

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(10,activation="sigmoid"))

model.compile(Adam(lr=0.001), loss="sparse_categorical_crossentropy" ,metrics=['accuracy'])
model.summary()

history_1 = model.fit(dataset_train_X,dataset_train_Y,batch_size=256,epochs=10,validation_data=[dataset_test_X,dataset_test_Y])

# Diffining Figure
f = plt.figure(figsize=(20,7))

#Adding Subplot 1 (For Accuracy)
f.add_subplot(121)

plt.plot(history_1.epoch,history_1.history['accuracy'],label = "accuracy") # Accuracy curve for training set
plt.plot(history_1.epoch,history_1.history['val_accuracy'],label = "val_accuracy") # Accuracy curve for validation set

plt.title("Accuracy Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Accuracy",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

#Adding Subplot 1 (For Loss)
f.add_subplot(122)

plt.plot(history_1.epoch,history_1.history['loss'],label="loss") # Loss curve for training set
plt.plot(history_1.epoch,history_1.history['val_loss'],label="val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()

plt.show()

print('Completed')
sound(5)

pickle.dump(model, open('MNIST.pkl', 'wb'))

model=pickle.load(open('MNIST.pkl', 'rb'))












