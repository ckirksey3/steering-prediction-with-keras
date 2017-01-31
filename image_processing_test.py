from model import process_img
from model import generate_arrays_from_lists
from model import get_lists_from_file
from model import image_pre_processing
from PIL import Image
import numpy as np

def display_processed_images():
    training_list = get_lists_from_file('data/driving_log.csv')
    for i in range(10):
        dataPoint = training_list[i]
        center_img_loc = dataPoint['center_img']
        steering_angle = dataPoint['steering_angle']
        image = Image.open(center_img_loc)
        image_array = np.asarray(image)

        image_array = image_pre_processing(image_array)
        image = Image.fromarray(np.uint8(image_array))
        image.show()


if __name__ == '__main__':
    display_processed_images()