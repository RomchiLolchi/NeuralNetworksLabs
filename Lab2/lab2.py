import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Задание 2 - зашумлённое изображение

if __name__ == "__main__":
    (og_train_images, og_train_labels), (og_test_images, og_test_labels) = mnist.load_data()

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype('float32') / 255

    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer="rmsprop",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    random_mnist_number_index = random.randint(0, len(og_test_images))
    random_test_image = og_test_images[random_mnist_number_index]
    random_test_label = og_test_labels[random_mnist_number_index]

    random_test_image_norm = random_test_image.reshape((1, 28 * 28))
    random_test_image_norm = random_test_image_norm.astype('float32') / 255

    plt.imshow(random_test_image)
    plt.show()

    signal = 0
    for pixel in range(28 * 28):
        signal += random_test_image_norm[0][pixel]

    # Отношение: полезный сигнал/сигнал с шумом
    ratio = []
    accuracy = []
    noice_epochs = 100
    max_delta = 1 / noice_epochs
    current_general_delta = 0
    for noice_epoch in range(noice_epochs):
        for pixel_width in range(28):
            for pixel_height in range(28):
                random_delta_norm = random.uniform(0, max_delta)
                current_general_delta += random_delta_norm
                random_test_image[pixel_width][pixel_height] += (random_delta_norm * 255)
                random_test_image_norm[0][pixel_width * pixel_height] += random_delta_norm
        if noice_epoch == 90:
            plt.imshow(random_test_image)
            plt.show()
        ratio.append(signal / (signal + current_general_delta))
        nn_prediction = model.predict(random_test_image_norm)
        print(f"Эпоха: {noice_epoch}, Наиболее вероятное: {nn_prediction[0].argmax()}")
        accuracy.append(nn_prediction[0][random_test_label])

    plt.scatter(ratio, accuracy)
    plt.show()
