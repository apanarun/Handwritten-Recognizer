import cv2
import pandas as pd
import numpy as np
import os

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import CSVLogger
from livelossplot import PlotLossesKeras

#Path
train_path,train_dirs,train_files = next(os.walk("Modified data/Train3"))
test_path,test_dirs,test_files = next(os.walk("Modified data/Validation3"))

#Creating Labels
labels = ['C', 'D', 'E', 'H', 'L', 'M', '0', 'P', 'R', 'T', 'U', 'W']
int_labels = [i for i in range(len(labels))]


#Converting Multiple CSV files into single csv file
train = pd.read_csv(train_path+'/C.csv')
test = pd.read_csv(test_path+'/C.csv')
y = train["label"]
y = np.full((y.shape[0],),int_labels[0])

for i in labels[1:]:
    temp_train = pd.read_csv(train_path+'/'+i+'.csv')
    train = train.append(temp_train)
    temp_y = temp_train["label"]
    y = np.append(y,np.full((temp_y.shape[0],),int_labels[labels.index(i)]))
    test = test.append(pd.read_csv(test_path+'/'+i+'.csv'))

print(train.shape,test.shape)

temp_test = test
x = train.drop(labels=["label"],axis=1)
test = test.drop(labels=["label"],axis=1)

#Constants
IMAGE_WIDTH = IMAGE_HEIGHT = 28
CHANNELS = 3
CLASSES = 13
VALIDATION_RATIO = 0.7
TRAINING_LOGS_FILE = os.getcwd()
EPOCHS = 1
BATCH_SIZE = 10
#VERBOSITY=1

#Reshaping
x = x.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
test = test.values.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
y = to_categorical(y, num_classes=CLASSES)
    
#Data Splitting
x_training, x_validation, y_training, y_validation = train_test_split(x, y, test_size=VALIDATION_RATIO, shuffle=True)

    
#Data Augmentation
data_generator = ImageDataGenerator(rescale=1./255,
                                    rotation_range=10,
                                    zoom_range=0.15,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1)
data_generator.fit(x_training)


#Building the model
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='Same',
                 activation='relu',
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
    
model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='Same',
                 activation='relu'))

    
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='Same',
                 activation='relu'))
    
model.add(Conv2D(filters=64,
                 kernel_size=(3,3),
                 padding='Same',
                 activation='relu'))
    
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(CLASSES, activation="softmax"))

model.compile(optimizer=RMSprop(lr=0.0001,
                                rho=0.9,
                                epsilon=1e-8,
                                decay=0.00001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])


#Training
history = model.fit_generator(data_generator.flow(x_training,
                                                  y_training,
                                                  batch_size=BATCH_SIZE),
                              epochs=EPOCHS,
                              validation_data=(x_validation, y_validation),
                              #verbose=VERBOSITY,
                              steps_per_epoch=x_training.shape[0],
                              callbacks=[PlotLossesKeras(),
                                         CSVLogger(TRAINING_LOGS_FILE,
                                                   append=False,
                                                   separator=";")])

#Testing
predictions = history.predict_classes(test, verbose=1)
pd.DataFrame({"ImageId":list(range(1,len(predictions)+1)),
              "Label":predictions}).to_csv("testoutputfile.csv")

#Saving the model
filename = "testmodel.exe"
joblib.dump(history,filename)


