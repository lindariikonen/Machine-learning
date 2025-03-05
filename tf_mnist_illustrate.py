import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from random import random


def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)

    correct = np.sum(pred == gt)
    accuracy = correct / len(gt) * 100

    return accuracy


# Original
mnist = tf.keras.datasets.mnist
# New
#mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
pred_random = np.random.randint(0, 10, size=y_test.shape)

# Print the size of training and test data
print(f'x_train shape {x_train.shape}')
print(f'y_train shape {y_train.shape}')
print(f'x_test shape {x_test.shape}')
print(f'y_test shape {y_test.shape}')

for i in range(x_test.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        accuracy = class_acc(pred_random, y_test)
        plt.figure(1)
        plt.clf()
        plt.imshow(x_test[i], cmap='gray_r')
        plt.title(f"Image {i} label num {y_test[i]} predicted {accuracy}")
        plt.pause(1)
