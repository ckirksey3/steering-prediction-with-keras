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
# import ipdb
import random
import argparse
import pickle

#Pre processing variables
BRIGHTNESS_RANGE = 10

# Model dimensions
INPUT_IMG_WIDTH = 64
INPUT_IMG_HEIGHT = 64
INPUT_CHANNELS = 3

# Angle to adjust by when switching to left/right img
ANGLE_ADJUSTMENT = 0.15

# Range of noise to add to steering angle
ANGLE_NOISE_MAX = 0.05

def image_pre_processing(img):
    # Add random brightness
    # Borrowed from Mohan Karthik's post (https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # brightness = BRIGHTNESS_RANGE + np.random.uniform()
    # img[:, :, 2] = img[:, :, 2] * brightness

    # Convert to grayscale
    # processed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Crop off top and bottom  of image to focus on road
    processed = img[60:140, 0:320]

    # Resize image
    processed = scipy.misc.imresize(processed, (INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT))

    return processed

# Add random noise to angle to help model avoid being stuck in a local minima
def add_random_noise_to_angle(steering_angle):
    return steering_angle + random.uniform(-1, 1) * ANGLE_NOISE_MAX

def choose_image_and_adjust_angle(center_img_loc, left_img_loc, right_img_loc, steering_angle):
    which_image = random.randint(0, 2)
    image_loc = center_img_loc
    if(which_image == 1):
        image_loc = left_img_loc
        steering_angle += ANGLE_ADJUSTMENT
    elif(which_image == 2):
        image_loc = right_img_loc
        steering_angle -= ANGLE_ADJUSTMENT
    return image_loc, steering_angle

def process_line(line):
    items = [x.strip() for x in line.split(',')]
    center_img_loc = items[0]
    left_img_loc = items[1]
    right_img_loc = items[2]
    steering_angle = float(items[3])
    steering_angle = add_random_noise_to_angle(steering_angle)
    return choose_image_and_adjust_angle(center_img_loc, left_img_loc, right_img_loc, steering_angle)

def process_img(image_array, add_dimension=False):
    image_array = image_pre_processing(image_array)
    if(add_dimension):
        image_array = image_array[None, :, :, None]
    else:
        image_array = image_array[None, :, :, :]
    image_array_flipped = np.fliplr(image_array)
    # change to this when sending to model.fit
    #image_array = image_array[:, :, None]
    return image_array, image_array_flipped

# Returns true if the angle is far enough away from zero using partially random threshold gets higher as training continues
# Adopted from Mohan Karthik's article (https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.304ci98i2)
def is_far_from_zero(angle, epoch):
    bias = 1. / (epoch + 1.)
    threshold = np.random.uniform()
    return ((abs(angle) + bias) > threshold)

def generate_arrays_from_file(path, use_batches=False, batch_size=10):
    f = open(path)
    while 1:
        if(use_batches):
            images = []
            angles = []
            current = 0
            for line in f:
                current += 1
                center_img, steering_angle = process_line(line)
                images.append(center_img)
                angles.append(steering_angle)
                if(current >= batch_size):
                    shuffle(images)
                    shuffle(angles)
                    yield (images, angles)
                    current = 0
                    images = []
                    angles = []
            f.close()
        else:
            for line in f:
                center_img, steering_angle = process_line(line)
                yield (center_img, steering_angle)
        f.close()
        

def get_lists_from_file(path):
    f = open(path)
    img_list = []
    angle_list = []
    for line in f:
        img_loc, steering_angle = process_line(line)
        image = Image.open(img_loc)
        image_array = np.asarray(image)
        transformed_image_array, flipped_transformed_image_array = process_img(image_array, add_dimension=False)
        img_list.append(transformed_image_array)
        angle_list.append(steering_angle)
    return img_list, angle_list

def get_list_from_file(path):
    f = open(path)
    training_list = []
    for line in f:
        center_img, steering_angle = process_line(line)
        training_list.append({'center_img': center_img, 'steering_angle': steering_angle})
    return training_list

def generate_arrays_from_lists(training_list, sample_size):
    i = 0
    while 1:
        for dataPoint in training_list:
            i += 1
            epoch = i/sample_size
            center_img_loc = dataPoint['center_img']
            steering_angle = dataPoint['steering_angle']
            if(is_far_from_zero(steering_angle, epoch)):
                image = Image.open(center_img_loc)
                image_array = np.asarray(image)
                
                transformed_image_array, flipped_transformed_image_array = process_img(image_array, add_dimension=False)
                yield(flipped_transformed_image_array, np.array([float(steering_angle) * -1.0]))
                yield(transformed_image_array, np.array([float(steering_angle)]))

def createNvidiaModel():
    col, row, ch = INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, INPUT_CHANNELS  # camera format

    #Create CNN based on NVIDIA paper (http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
    model = Sequential()
    #Add lambda function to avoid pre-processing outside of network
    model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(col, row, ch),
        output_shape=(col, row, ch)))
    # model.add(Reshape((66, 200, 3), input_shape=(160, 320, 3)))
    #Deviates from NVIDIA paper
    model.add(Conv2D(3, 5, 5, input_shape=(80, 160, 3), subsample=(2, 2), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(3, 5, 5, input_shape=(80, 160, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(24, 5, 5, input_shape=(31, 98, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(36, 5, 5, input_shape=(5, 22, 3), subsample=(2, 2), activation='relu'))
    model.add(Conv2D(48, 3, 3, input_shape=(3, 20, 3), activation='relu'))
    model.add(Conv2D(64, 3, 3, input_shape=(1, 18, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(Dense(1164, activation='linear'))
    model.add(Dropout(.5))
    model.add(Dense(100, activation='linear'))
    model.add(Dense(50, activation='linear'))
    model.add(Dense(1, activation='linear'))
    model.summary()
    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Save model to json file
    json_string = model.to_json()
    with open('model.json','w') as f:
            json.dump(json_string,f)
    return model

def createOldModel():
    col, row, ch = INPUT_IMG_WIDTH, INPUT_IMG_HEIGHT, 1  # camera format

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


def initialize(training_type, model=createNvidiaModel(), plot=False, validation_split=0.5):
    if(training_type == "test_with_3"):
        img_list, angle_list = get_lists_from_file('test_driving_log.csv')
        history =model.fit(np.array(img_list), np.array(angle_list),
            batch_size=12, nb_epoch=50, validation_split=0.5, shuffle=True)
        # from Jason Brownlee http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        model.save_weights("model.h5")
        return model
    else:
        training_list = []
        if(training_type == "full"):
            training_list = get_list_from_file('data/driving_log.csv')
        elif(training_type == "full_with_less_zeros"):
            training_list = get_list_from_file('data/driving_log_less_zeros.csv')
        elif(training_type == "test_with_3_generator"):
            training_list = get_list_from_file('test_driving_log.csv')

        list_length = len(training_list)

        np.random.shuffle(training_list)
        train_generator = generate_arrays_from_lists(training_list, list_length)

        np.random.shuffle(training_list)
        validation_generator = generate_arrays_from_lists(training_list, list_length)

        nb_train_samples = int(list_length * validation_split)
        nb_val_samples = list_length - nb_train_samples

        history = model.fit_generator(train_generator, samples_per_epoch=list_length, nb_epoch=30, validation_data=validation_generator, nb_val_samples=nb_val_samples)
        
        # Save history and weights
        pickle.dump(history.history, open('history.p', 'wb'))
        model.save_weights("model.h5")

        # Plot loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training driving model')
    parser.add_argument('training_mode', type=str,
        help='Training mode type (test_with_3, full, full_with_less_zeros, test_from_less_zeros)')
    parser.add_argument('plot', type=str, nargs='?', default=False,
        help='Whether to display plot of loss after training.')
    parser.add_argument('model', type=str, nargs='?', default=None,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    gc.collect()

    if args.model is not None:
        with open(args.model, 'r') as jfile:
            model = model_from_json(json.load(jfile))
            model.compile("adam", "mse")
            weights_file = args.model.replace('json', 'h5')
            model.load_weights(weights_file)
            initialize(args.training_mode, model, args.plot)
    else:
        initialize(args.training_mode)
    
#TODO implement transfer learning from previous successful models