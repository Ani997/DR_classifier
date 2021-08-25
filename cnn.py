from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K    

# dimensions of our images
img_width, img_height = 300, 300                                # Height and width of input image

train_data_dir = 'D:/dog_cat_classifier/dataset/train'            # Training data directory
validation_data_dir = 'D:/dog_cat_classifier/dataset/test'    # Validation data directory
nb_train_samples = 1600              # No of Training samples
nb_validation_samples = 398          # No of Validation samples
epochs = 10                          # No of epochs
batch_size = 10                      # No of samples to be passed to NN

if K.image_data_format() == 'channels_first':         # Checking the image format
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(rescale=1. / 255,                               # rescaling factor 
                                   shear_range=0.2,                                # shear intensity
                                   zoom_range=0.2,                                 # range for random zoom
                                   horizontal_flip=True)                           # randomly flips inputs horizontally

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                        target_size=(img_width, img_height),
                                                        batch_size=batch_size,
                                                        class_mode='binary')

model = Sequential()
model.add(Conv2D(128, (3, 3), input_shape=input_shape))         # input layer,128 filters, 3x3 kernel size
model.add(Activation('relu'))                                  # using relu activation function
model.add(MaxPooling2D(pool_size=(2,2)))                       # max pooling                                           

model.add(Conv2D(32, (3, 3)))                                  # 1st hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3, 3)))                                  # 2nd hidden layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())                                           # flattening the model
model.add(Dense(32))                                           
model.add(Activation('relu'))
model.add(Dropout(0.5))                                        # adding dropout
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint = ModelCheckpoint("D:/dog_cat_classifier/model.h5",
                             monitor='val_accuracy', verbose=1,
                             save_best_only=True, save_weights_only=False,
                             mode='auto', period=1)

early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=10,
                      verbose=1, mode='auto')

hist = model.fit_generator(steps_per_epoch=nb_train_samples // batch_size,
                           generator=train_generator, validation_data=validation_generator,
                           validation_steps=nb_validation_samples // batch_size, 
                           epochs=epochs, callbacks=[checkpoint,early])

import matplotlib.pyplot as plt
plt.plot(hist.history["accuracy"])
plt.plot(hist.history["val_accuracy"])
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epochs")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation loss"])
plt.show()
