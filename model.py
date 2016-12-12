from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten

# Normalization
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_train = X_train / 255 - 0.5
X_val = X_val / 255 - 0.5

Y_train = np_utils.to_categorical(y_train, 43)
Y_val = np_utils.to_categorical(y_val, 43)


model = Sequential()
model.add(Conv2D(3, 5, 5, input_shape=(66, 200, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(24, 5, 5, input_shape=(31, 98, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, input_shape=(5, 22, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 3, 3, input_shape=(3, 20, 3), activation='relu'))
model.add(Conv2D(64, 3, 3, input_shape=(1, 18, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
# TODO: Compile and train the model here.
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val, Y_val))

X_train_flat = X_train.reshape(-1, 32*32*3)
X_val_flat = X_val.reshape(-1, 32*32*3)

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# def generate_arrays_from_file(path):
#     while 1:
#     f = open(path)
#     for line in f:
#         # create numpy arrays of input data
#         # and labels, from each line in the file
#         x1, x2, y = process_line(line)
#         yield ({'input_1': x1, 'input_2': x2}, {'output': y})
#     f.close()

# model.fit_generator(generate_arrays_from_file('/my_file.txt'),
#         samples_per_epoch=10000, nb_epoch=10)

#TODO implement transfer learning from previous successful models
#TODO get at least 40k samples
#TODO Copy the Nvidia pipeline
#TODO Re-size the input image down by 2
#TODO use 5 epochs

history = model.fit(X_train_flat, Y_train,
                    batch_size=128, nb_epoch=20,
                    verbose=1, validation_data=(X_val_flat, Y_val))