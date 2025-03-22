import sys
import numpy as np
from PIL import Image
from network import NeuralNetwork

def load_image(image_path):
    """ Завантажуємо зображення правильно у бінарному режимі """
    with open(image_path, "rb") as f:  # Відкриваємо як бінарний файл
        image = Image.open(f).convert('L')  # Конвертуємо в градації сірого
        image = image.resize((6, 6))  # Масштабуємо до 6x6
        return np.array(image).flatten()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use: python predict_image.py <image_path.png>")
        sys.exit(1)


    label_dict = {
        "vertical_line": [0, 0],
        "horizontal_line": [1, 0],
        "two_vertical_lines": [0, 1],
        "two_horizontal_lines": [1, 1]
    }

    label_names = list(label_dict.keys())
    label_values = np.array(list(label_dict.values()))

    image_path = sys.argv[1]
    nn = NeuralNetwork(config_file="nn_model.json")

    image = load_image(image_path)

    output = nn.predict(image)

    # Знаходимо найближчий клас за найменшою евклідовою відстанню
    distances = np.linalg.norm(label_values - output, axis=1)
    predicted_label_index = np.argmin(distances)
    predicted_label = label_names[predicted_label_index]

    # результат
    print(f"Прогнозовані значення: {output}")
    print(f"Прогнозований клас: {predicted_label}")
