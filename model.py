from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten, Reshape, Lambda, Dropout
from keras.layers import Convolution2D, ELU
from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
import json
from PIL import Image
import io, base64
import numpy as np
import scipy.misc
from matplotlib import pyplot as plt
import cv2
import gc
import ipdb
import matplotlib.pyplot as plt

#Train model

def image_pre_processing(img):
    # 1) Convert to grayscale
    processed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Apply perspective transform
    # img_size = (img.shape[1], img.shape[0])
    # src = np.float32([(0, 65), (0,320), (160, 320), (160, 65)])
    # dst = np.float32([(0, 0), (0,320), (160, 320), (160, 0)])
    # M = cv2.getPerspectiveTransform(src, dst)
    # processed = cv2.warpPerspective(processed, M, img_size)

    # 3) Crop off top of image
    processed = processed[60:160, 0:320]
    return processed

def process_line(line, asList):
    items = [x.strip() for x in line.split(',')]
    center_img_loc = items[0]
    steering_angle = items[3]
    # steering_angle = (float(items[3]) + 1.0)/2.0
    if(asList):
        return center_img_loc, np.array([steering_angle])
    else:
        return center_img_loc, steering_angle

def process_img(image_array):
    image_array = image_pre_processing(image_array)
    image_array = scipy.misc.imresize(image_array, (50, 160))
    image_array = image_array[None, :, :, None]
    # change to this when sending to model.fit
    #image_array = image_array[:, :, None]
    return image_array

def generate_arrays_from_file(path):
    # f = open(path)
    # while 1:
    #     images = []
    #     angles = []
    #     current = 0
    #     for line in f:
    #         current += 1
    #         center_img, steering_angle = process_line(line)
    #         images.append(center_img)
    #         angles.append(steering_angle)
    #         if(current >= batch_size):
    #             shuffle(images)
    #             shuffle(angles)
    #             yield (images, angles)
    #             current = 0
    #             images = []
    #             angles = []
    #     f.close()
    while 1:
        f = open(path)
        for line in f:
            center_img, steering_angle = process_line(line, True)
            yield (center_img, steering_angle)
        f.close()

def get_lists_from_file(path):
    f = open(path)
    training_list = []
    for line in f:
        center_img, steering_angle = process_line(line, True)
        training_list.append({'center_img': center_img, 'steering_angle': steering_angle})
    return training_list

def generate_arrays_from_lists(training_list):
    while 1:
        for dataPoint in training_list:
            center_img_loc = dataPoint['center_img']
            steering_angle = dataPoint['steering_angle']
            image = Image.open(center_img_loc)
            image_array = np.asarray(image)
            
            # ipdb.set_trace()
            transformed_image_array = process_img(image_array)
            yield (transformed_image_array, steering_angle)

def createModel():
    #Create CNN based on NVIDIA paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    # model = Sequential()
    # #Add lambda function to avoid pre-processing outside of network
    # model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(80, 160, 3),
    #         output_shape=(80, 160, 3)))
    # # model.add(Reshape((66, 200, 3), input_shape=(160, 320, 3)))
    # #Deviates from NVIDIA paper
    # model.add(Conv2D(3, 5, 5, input_shape=(80, 160, 3), subsample=(2, 2), activation='relu'))
    # # model.add(BatchNormalization())
    # # model.add(Conv2D(3, 5, 5, input_shape=(80, 160, 3), subsample=(2, 2), activation='relu'))
    # model.add(Conv2D(24, 5, 5, input_shape=(31, 98, 3), subsample=(2, 2), activation='relu'))
    # model.add(Conv2D(36, 5, 5, input_shape=(5, 22, 3), subsample=(2, 2), activation='relu'))
    # model.add(Conv2D(48, 3, 3, input_shape=(3, 20, 3), activation='relu'))
    # model.add(Conv2D(64, 3, 3, input_shape=(1, 18, 3), activation='relu'))
    # model.add(Flatten())
    # model.add(Dropout(.2))
    # model.add(Dense(1164, activation='linear'))
    # model.add(Dropout(.5))
    # model.add(Dense(100, activation='linear'))
    # model.add(Dense(50, activation='linear'))
    # model.add(Dense(1, activation='linear'))
    # model.summary()
    # model.compile(loss='mean_squared_error',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    col, row, ch = 50, 160, 1  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
            input_shape=(col, row, ch),
            output_shape=(col, row, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    # Save model to json file
    json_string = model.to_json()
    with open('model.json','w') as f:
            json.dump(json_string,f)
    return model


def initialize():
    training_list = get_lists_from_file('test_driving_log.csv')
    np.random.shuffle(training_list)
    model = createModel()
    # ipdb.set_trace()
    # history =model.fit(np.array(img_list), np.array(angle_list),
    #     batch_size=12, nb_epoch=50, validation_split=0.5, shuffle=True)
    history = model.fit_generator(generate_arrays_from_lists(training_list),
        samples_per_epoch=12, nb_epoch=15)
    # model.fit_generator(generate_arrays_from_file('data/driving_log.csv'),
    #     samples_per_epoch=10000, nb_epoch=5)
    # model.fit_generator(generate_arrays_from_file('driving_log.csv'),
    #         samples_per_epoch=1000, nb_epoch=15)
    # model.fit_generator(generate_arrays_from_file('test_driving_log.csv'),
    #     samples_per_epoch=12, nb_epoch=15)
    model.save_weights("model.h5")
    # from Jason Brownlee http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    return model

if __name__ == '__main__':
    gc.collect()
    initialize()
#TODO implement transfer learning from previous successful models