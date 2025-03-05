import matplotlib.pyplot as plt
from keras import Sequential
from keras.api.datasets import mnist
from keras.src.layers import Dense
from keras.src.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
x_test = x_test.reshape(-1, 784).astype('float32') / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Dense(5, input_dim=784, activation='sigmoid'))  # Hidden layer with 5 neurons
model.add(Dense(10, activation='softmax'))  # Output layer with 10 neurons (one for each class)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

train_loss, train_acc = model.evaluate(x_train, y_train)
test_loss, test_acc = model.evaluate(x_test, y_test)

print(f"Training Accuracy: {train_acc * 100:.2f}%")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
