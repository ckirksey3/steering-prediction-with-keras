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
ANGLE_ADJUSTMENT = 0.25

# Range of noise to add to steering angle
ANGLE_NOISE_MAX = 0.05

# Min bucket size for steering angle distribution
MIN_BUCKET_SIZE = 2

# PERCENTAGE OF DATA TO BE USED FOR TRAINING INSTEAD OF VALIDATION
VALIDATION_SPLIT = 0.8

# Breaks input into buckets based on steering angle
def segment_data_by_angles(angles, images):
    counts, bucket_indices = np.histogram(angles, bins='auto')
    bucket_increment = bucket_indices[1] - bucket_indices[0]
    bucket_min = bucket_indices[0]
    buckets = [list() for _ in range(len(counts))]
    # TODO adjust for flipping images
    for i in range(len(angles)):
        img = images[i]
        angle = angles[i]
        bucket_index = int((angle - bucket_min)/bucket_increment)
        bucket_index = min(bucket_index, len(buckets) - 1)
        bucket_index = max(bucket_index, 0)
        buckets[bucket_index].append({'img': img, 'angle': angle})
    merged_buckets = []
    merged_buckets.append(buckets[0])
    for i in range(1, len(buckets)):
        if(len(buckets[i]) < MIN_BUCKET_SIZE):
            merged_buckets[-1] += buckets[i]
        else:
            merged_buckets.append(buckets[i])
    return merged_buckets

# Random shadow
# Created by Vivek Yadav (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.fwuosa9qd)
def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        # random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def image_pre_processing(img):
    #Add random shadow
    img = add_random_shadow(img)

    # Add random brightness
    # Borrowed from Mohan Karthik's post (https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff)
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    bright = .25+np.random.uniform()
    img[:,:,2] = img[:,:,2]*bright
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

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

def process_line(line, direction):
    items = [x.strip() for x in line.split(',')]
    img_loc = items[direction]
    steering_angle = float(items[3])
    steering_angle = add_random_noise_to_angle(steering_angle)
    if(direction == 1):
        steering_angle += ANGLE_ADJUSTMENT
    elif(direction == 2):
        steering_angle -= ANGLE_ADJUSTMENT
    return img_loc, steering_angle

def process_img(image_array, add_dimension=False):
    image_array = image_pre_processing(image_array)
    if(add_dimension):
        image_array = image_array[:, :, None]
    else:
        image_array = image_array[:, :, :]
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

# Image shifts
# Created by Vivek Yadav (https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.fwuosa9qd)
def trans_image(image,steer,trans_range):
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1], image.shape[0]))
    return image_tr,steer_ang        


def get_lists_from_file(path):
    f = open(path)
    img_list = []
    angle_list = []
    for line in f:
        for direction in range(2):
            img_loc, steering_angle = process_line(line, direction)
            img_list.append(img_loc)
            angle_list.append(steering_angle)
    return img_list, angle_list

def generate_arrays_from_lists(data_buckets, sample_size, batch_size=1):
    i = 0
    image_list = []
    angle_list = []
    index_within_each_list = 0
    while 1:
        for training_list in data_buckets:
            bucket_length = len(training_list)
            dataPoint = training_list[index_within_each_list % bucket_length]
            i += 2
            epoch = i/sample_size
            img_loc = dataPoint['img']
            steering_angle = dataPoint['angle']
            if(is_far_from_zero(steering_angle, epoch)):
                image = Image.open(img_loc)
                image_array = np.asarray(image)
                
                transformed_image_array, flipped_transformed_image_array = process_img(image_array, add_dimension=False)

                # Add original
                image_list.append(transformed_image_array)
                angle_list.append(float(steering_angle))

                # Add Flipped
                image_list.append(flipped_transformed_image_array)
                angle_list.append(float(steering_angle) * -1.0)

                if(len(image_list) >= batch_size):
                    yield(np.array(image_list), np.array(angle_list))
                    image_list = []
                    angle_list = []
        index_within_each_list += 1

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

def initialize(data_buckets, model=None):
        if(model is None):
            model = createNvidiaModel()

        # Adjust sample size to account for horizontal flipping in image processing
        sample_size = 25000
        batch_size = 64
        nb_epoch = 30
        train_generator = generate_arrays_from_lists(data_buckets, sample_size, batch_size=batch_size)
        validation_generator = generate_arrays_from_lists(data_buckets, sample_size, batch_size=batch_size)

        nb_train_samples = int(sample_size * VALIDATION_SPLIT)
        print("nb_train_samples", nb_train_samples)
        nb_val_samples = sample_size - nb_train_samples
        print("nb_val_samples", nb_val_samples)
        history = model.fit_generator(train_generator, samples_per_epoch=sample_size, nb_epoch=nb_epoch, validation_data=validation_generator, nb_val_samples=nb_val_samples)
        
        # Save history and weights
        pickle.dump(history.history, open('history.p', 'wb'))
        model.save_weights("model.h5")

        gc.collect()
        return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training driving model')
    parser.add_argument('model', type=str, nargs='?', default=None,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    gc.collect()

    img_list, angle_list = get_lists_from_file('data/driving_log.csv')
    data_buckets = segment_data_by_angles(angle_list, img_list)

    if args.model is not None:
        with open(args.model, 'r') as jfile:
            model = model_from_json(json.load(jfile))
            model.compile("adam", "mse")
            weights_file = args.model.replace('json', 'h5')
            model.load_weights(weights_file)
            initialize(data_buckets, model)
    else:
        initialize(data_buckets)