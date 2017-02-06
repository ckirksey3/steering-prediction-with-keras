from model import process_img
from model import generate_arrays_from_lists
from model import get_lists_from_file
from model import image_pre_processing
from model import trans_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

def display_processed_img(image_loc):
    image = Image.open(image_loc)
    image_array = np.asarray(image)
    image_array = image_pre_processing(image_array)
    image = Image.fromarray(np.uint8(image_array))
    image.show()

def display_processed_images():
    training_list = get_lists_from_file('data/driving_log.csv')
    for i in range(10):
        dataPoint = training_list[i]
        center_img_loc = dataPoint['center_img']
        display_processed_img(center_img_loc)

def simulate_axis_flip(angle_list):
    result = []
    for angle in angle_list:
        result.append(angle)
        result.append(angle * -1.0)
    return result

def simulate_image_shifting(angle_list):
    result = []
    TRANS_ANGLE = 0.1
    i = 0
    total_added  = 0
    for angle in angle_list:
        i += 1
        result.append(angle)
        amount_to_add = TRANS_ANGLE * (2.0 * np.random.uniform() - 1)
        total_added += amount_to_add
        result.append(angle + amount_to_add)
    print("Average added by image shifting: ", total_added/i)
    return result

def simulate_camera_switching(angle_list):
    ANGLE_ADJUSTMENT = 0.15
    result = []
    for angle in angle_list:
        result.append(angle + ANGLE_ADJUSTMENT * (np.random.randint(3) - 1.0))
    return result

def simulate_penelize_zeros(angle_list):
    result = []
    i = 0
    for angle in angle_list:
        i += 1
        cur_round = i / (len(angle_list)/20)
        bias = 1. / (cur_round + 1.)
        threshold = np.random.uniform()
        if (abs(angle) + bias) > threshold:
            result.append(angle)
    return result

def display_angle_distribution():
    print("Processing file..")
    img_list, angle_list = get_lists_from_file('data/2.4_recording_dirt_turn/driving_log.csv')
    print("Finished")
    angle_list = simulate_camera_switching(angle_list)
    angle_list = simulate_axis_flip(angle_list)
    angle_list = simulate_image_shifting(angle_list)
    angle_list = simulate_penelize_zeros(angle_list)
    plt.hist(angle_list, bins="auto")
    plt.title("Angle Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.show()

if __name__ == '__main__':
    display_angle_distribution()
    # for i in range(5):
    #     display_processed_img("IMG/center_2016_12_01_13_31_15_005.jpg")