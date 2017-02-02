import argparse
import base64
import json
import cv2
from PIL import Image

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from model import process_img
from model import initialize
from model import get_list_from_file
from model import image_pre_processing
from data_analysis import display_processed_img

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    # parser.add_argument('image', type=str,
    # help='Image to test model on')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    # model = initialize()

    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    images = [["/Applications/IMG/center_2016_12_11_10_23_45_860.jpg", -0.3, -0.06, -0.1227381], ["/Applications/IMG/center_2016_12_11_10_23_59_993.jpg", -0.1, 0.1, 0], ["/Applications/IMG/center_2016_12_11_10_24_57_201.jpg", 0.1, 0.6, 0.3380598]]
    for test_image in images:
        image = Image.open(test_image[0])
        image_array = np.asarray(image)
        transformed_image_array, flipped_transformed_image_array = process_img(image_array, False)

        steering_angle = model.predict(np.array([transformed_image_array]), batch_size=1)
        # display_processed_img(test_image[0])
        if(steering_angle > test_image[1] and steering_angle < test_image[2]):
            print("Passed")
        else:
            print("Failed")
            print("Target: ", test_image[3])
            print("Actual: ", steering_angle)

    # training_list = get_list_from_file('data/driving_log_less_zeros.csv')
    # for dataPoint in training_list[:50]:
    #         center_img_loc = dataPoint['center_img']
    #         expected_steering_angle = float(dataPoint['steering_angle'])
    #         image = Image.open(center_img_loc)
    #         image_array = np.asarray(image)
    #         transformed_image_array, flipped_transformed_image_array = process_img(image_array)

    #         # steering_angle = (float(model.predict(transformed_image_array, batch_size=1)) * 2.0) - 1.0
    #         steering_angle = float(model.predict(np.array(transformed_image_array), batch_size=1))
    #         print("***********")
    #         print("Dif: ", abs(expected_steering_angle - steering_angle))
    #         print("Target: ", expected_steering_angle)
    #         print("Actual: ", steering_angle)
