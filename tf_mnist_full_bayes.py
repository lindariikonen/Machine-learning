import tensorflow as tf
import numpy as np
from scipy.stats import multivariate_normal


def class_acc(pred, gt):
    pred = np.array(pred)
    gt = np.array(gt)

    correct = np.sum(pred == gt)
    accuracy = correct / len(gt) * 100

    return accuracy


def log_likelihood(x, mean, var):
    return -0.5 * np.sum(np.log(2.0 * np.pi * var) + ((x - mean) ** 2) / var, axis=1)


def predict_full_bayes(x_test):
    log_probs = []
    for i in range(10):
        log_prob = multivariate_normal.logpdf(x_test, mean=means[i], cov=covariances[i])
        log_probs.append(log_prob)
    return np.argmax(log_probs, axis=0)


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

scale = [0.1, 1.0, 10.0]

for sc in scale:
    means = []
    covariances = []
    noise = np.random.normal(loc=0.0, scale=sc, size=x_train_flat.shape)
    x_train_flat_noisy = x_train_flat + noise

    for i in range(10):
        class_data = x_train_flat_noisy[y_train == i]
        means.append(np.mean(class_data, axis=0))
        covariances.append(np.var(class_data, axis=0))

    means = np.array(means)
    covariances = np.array(covariances)

    y_pred = predict_full_bayes(x_test_flat)
    accuracy = class_acc(y_pred, y_test)
    print(f'Classification accuracy for scale {sc:.1f}: {accuracy:.2f}')
