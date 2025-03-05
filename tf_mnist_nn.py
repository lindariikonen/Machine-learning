import tensorflow as tf
import numpy as np
import sys
from sklearn.neighbors import KNeighborsClassifier


def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)

    correct = np.sum(pred == gt)
    accuracy = correct / len(gt) * 100

    return accuracy


#input = sys.argv[1]
# couldn't get tensorflow to load via consol, but it works in IDE
input = input("Write original or fashion to choose dataset: ")

if input == "original":
    mnist = tf.keras.datasets.mnist
elif input == "fashion":
    mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train_flat, y_train)

y_pred = knn.predict(x_test_flat)
accuracy = class_acc(y_pred, y_test)
print(f'Classification accuracy: {accuracy:.2f}')
