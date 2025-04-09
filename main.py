import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class BColors:
    HEADER = '\033[95m'
    OkBLUE = '\033[94m'
    OkCYAN = '\033[96m'
    OkGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class ImageUtils:
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load an image from file as a numpy array."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        return np.array(Image.open(filepath))

    @staticmethod
    def save_image(image_array: np.ndarray, filepath: str) -> None:
        """Save a numpy array as an image file."""
        if image_array.dtype != np.uint8:
            image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        Image.fromarray(image_array).save(filepath)

    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """Normalize image to range [0, 255]."""
        min_val = np.min(image_array)
        max_val = np.max(image_array)

        if max_val > min_val:
            normalized = (image_array - min_val) / (max_val - min_val) * 255
            return normalized
        else:
            return np.zeros_like(image_array)


class ImageFilter:
    """Class for different image filtering operations."""
    pass


class Visualizer:
    """Class for visualizing image processing results."""
    pass


def main():

    base_dir = os.path.join("Images")
    os.makedirs(base_dir, exist_ok=True)

    # ======================================================
    # Create Gaussian Filter
    # ======================================================



if __name__ == '__main__':
    main()