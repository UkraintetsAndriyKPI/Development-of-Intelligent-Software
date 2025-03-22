import os
import matplotlib.pyplot as plt
import numpy as np

def generate_and_save_images():
    images = {
        "vertical_line": np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ]),
        "horizontal_line": np.array([
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ]),
        "two_vertical_lines": np.array([
            [0, 1, 0, 0, 1, 0,],
            [0, 1, 0, 0, 1, 0,],
            [0, 1, 0, 0, 1, 0,],
            [0, 1, 0, 0, 1, 0,],
            [0, 1, 0, 0, 1, 0,],
            [0, 1, 0, 0, 1, 0,]
        ]),
        "two_horizontal_lines": np.array([
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0]
        ])
    }

    os.makedirs("training_images", exist_ok=True)

    for name, image in images.items():
        plt.imshow(image, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.savefig(f"training_images/{name}.png")
        plt.close()


generate_and_save_images()
