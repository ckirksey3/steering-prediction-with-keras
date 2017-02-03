import numpy as np
from matplotlib import pyplot as plt
import gc
import argparse
import pickle

def plot_history(history):
    with open(history, "rb") as history_file:
        history_obj = pickle.load(history_file)
    plt.plot(history_obj['loss'])
    plt.plot(history_obj['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot history')
    parser.add_argument('history', type=str, help='Path to pickle file where history of model training is saved.')
    args = parser.parse_args()
    gc.collect()
    plot_history(args.history)