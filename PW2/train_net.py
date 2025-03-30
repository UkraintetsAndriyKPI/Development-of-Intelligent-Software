import os
import numpy as np
from PIL import Image

from network import NeuralNetwork


def load_images_and_labels(image_folder="train_images"):
    images = []
    labels = []
    label_dict = {
        "vertical_line": [0, 0],
        "horizontal_line": [1, 0],
        "two_vertical_lines": [0, 1],
        "two_horizontal_lines": [1, 1]
    }

    for filename in os.listdir(image_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)

            image = Image.open(image_path).convert('L')  # Конвертуємо в відтінки сірого
            image = image.resize((6, 6))

            image_array = np.array(image).flatten()

            label = filename.split(".")[0]

            labels.append(label_dict[label])
            images.append(image_array)

    # print(images)
    # print(labels)
    return images, labels


def train_network():
    train_X, train_Y = load_images_and_labels()
    train_X, train_Y = np.array(train_X), np.array(train_Y)

    y_one_hot = np.array([label for label in train_Y])
    # print(y_one_hot)

    # Створюємо нейронну мережу
    input_size = train_X.shape[1]  # 6x6, 36 нейронів у вхідному шарі
    hidden_sizes = [36, 36]  # Прихований шар нейронів
    output_size = y_one_hot.shape[1]  # (2 нейрони на виході)
    network = NeuralNetwork(input_size, hidden_sizes, output_size)


    network.train(train_X, y_one_hot, epochs=100000)

    # Зберігаємо модель
    network.save_model("nn_model.json")

if __name__ == '__main__':
    train_network()
