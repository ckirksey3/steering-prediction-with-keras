from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten, Reshape
from keras.models import model_from_json
import json

# Normalization
# X_train = X_train.astype('float32')
# X_val = X_val.astype('float32')
# X_train = X_train / 255 - 0.5
# X_val = X_val / 255 - 0.5

#Create CNN based on NVIDIA paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
model = Sequential()
# model.add(Reshape((66, 200, 3), input_shape=(160, 320, 3)))
#Deviates from NVIDIA paper
# model.add(Conv2D(3, 5, 5, input_shape=(66, 200, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(3, 5, 5, input_shape=(160, 320, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(24, 5, 5, input_shape=(31, 98, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, input_shape=(5, 22, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 3, 3, input_shape=(3, 20, 3), activation='relu'))
model.add(Conv2D(64, 3, 3, input_shape=(1, 18, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(1164, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Save model to json file
json_string = model.to_json()
with open('model.json','w') as f:
        json.dump(json_string,f)

#Train model
imgString = data["image"]
image = Image.open(BytesIO(base64.b64decode(imgString)))
image_array = np.asarray(image)
transformed_image_array = image_array[None, :, :, :]
# history = model.fit(X_train_flat, Y_train,
#                     batch_size=128, nb_epoch=5,
#                     verbose=1, validation_data=(X_val_flat, Y_val))

def process_line(line):
    

def generate_arrays_from_file(path):
    while 1:
    f = open(path)
    for line in f:
        # create numpy arrays of input data
        # and labels, from each line in the file
        x1, x2, y = process_line(line)
        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
    f.close()

model.fit_generator(generate_arrays_from_file('driving_log.csv'),
        samples_per_epoch=10000, nb_epoch=10)

model.save_weights("model.h5")
#TODO implement transfer learning from previous successful models
#TODO get at least 40k samples (currently 1.3K)
#TODO Re-size the input image down by 2
