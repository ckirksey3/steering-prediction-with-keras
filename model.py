from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten, Reshape
from keras.models import model_from_json
import json
from PIL import Image
import io, base64
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt

#Create CNN based on NVIDIA paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
model = Sequential()
# model.add(Reshape((66, 200, 3), input_shape=(160, 320, 3)))
#Deviates from NVIDIA paper
# model.add(Conv2D(3, 5, 5, input_shape=(66, 200, 3), subsample=(2, 2), activation='relu'))
model.add(Conv2D(3, 5, 5, input_shape=(80, 160, 3), subsample=(2, 2), activation='relu'))
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
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

# Save model to json file
json_string = model.to_json()
with open('model.json','w') as f:
        json.dump(json_string,f)

#Train model
def process_line(line):
    items = [x.strip() for x in line.split(',')]
    center_img_loc = items[0]
    image = Image.open(center_img_loc)
    image_array = np.asarray(image)
    transformed_image_array = process_img(image_array)
    return transformed_image_array, np.array([items[3]])

def process_img(image_array):
    image_array = scipy.misc.imresize(image_array, (80, 160))
    return image_array[None, :, :, :]

def generate_arrays_from_file(path):
    while 1:
        f = open(path)
        for line in f:
        # for i in range(5):
        #     line = f.readline()
            # create numpy arrays of input data
            # and labels, from each line in the file
            center_img, steering_angle = process_line(line)
            # print(center_img.shape, steering_angle)
            # plt.imshow(center_img.squeeze(), interpolation='nearest')
            # plt.show()
            yield (center_img, steering_angle)
        f.close()
# generate_arrays_from_file('test_driving_log.csv')
model.fit_generator(generate_arrays_from_file('driving_log.csv'),
        samples_per_epoch=1000, nb_epoch=5)

model.save_weights("model.h5")
#TODO implement transfer learning from previous successful models
#TODO get at least 40k samples (currently 1.3K)
