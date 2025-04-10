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

    @staticmethod
    def center_domain(image_array: np.ndarray) -> np.ndarray:
        y_grid, x_grid = np.mgrid[0:image_array.shape[0], 0:image_array.shape[1]]
        mask = np.where( (x_grid + y_grid) % 2 == 0, 1, -1)
        return image_array * mask

class ImageFilter:
    """Class for different image filtering operations."""
    @staticmethod
    def create_ideal_low_pass_filter(size: Tuple[int, int], radius: float) -> np.ndarray:
        """Create an ideal low pass filter."""
        y_grid, x_grid = np.mgrid[0:size[0], 0:size[1]]
        center_y, center_x = size[0] // 2, size[1] // 2

        distance_sq = (y_grid - center_y) ** 2 + (x_grid - center_x) ** 2
        lpf = np.where(distance_sq <= radius ** 2, 255, 0)
        return lpf

class Visualizer:
    """Class for visualizing image processing results."""
    @staticmethod
    def display_images(
            images: list,
            titles: list,
            filepath: str = None,
            cols: int = 3,
            cmap: str = 'gray',
            fig_size: Tuple[int, int] = (15, 5)
    ) -> None:
        """Display multiple images in a single figure with titles."""
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=fig_size)
        # Flatten in case of multiple axes; if a single axis, wrap it into list.
        if rows * cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title)
            ax.axis("off")
        # Turn off any extra axes if available.
        for ax in axes[n_images:]:
            ax.axis("off")
        plt.tight_layout()
        if filepath is not None:
            plt.savefig(filepath)
        plt.show()


def main():

    base_dir = os.path.join("Images")
    os.makedirs(base_dir, exist_ok=True)

    radius = 60
    # ======================================================
    # Creating Ideal Low Pass Filter (LPF in Transform Domain)
    # ======================================================
    print("=" * 50)
    print(f"{BColors.WARNING}{BColors.BOLD}"
          f"Creating Ideal Low Pass Filter (LPF in Transform Domain)......"
          f"{BColors.ENDC}{BColors.ENDC}")
    lpf_transform_domain = ImageFilter.create_ideal_low_pass_filter(
        size=(1024, 1024),
        radius=radius,
    )
    print(f"\nIdeal Low Pass Filter Array (Transform Domain) (radius={radius})\n", lpf_transform_domain)
    ImageUtils.save_image(
        image_array=lpf_transform_domain,
        filepath=os.path.join(base_dir, f"lpf(r={radius})_transform_domain.png")
    )
    print(f"{BColors.OkGREEN}{BColors.BOLD}"
          f"\nCreating Ideal Low Pass Filter (LPF in Transform Domain) Finished."
          f"{BColors.ENDC}{BColors.ENDC}")

    # ======================================================
    # Centering The Ideal Low Pass Filter in the Spatial Domain (LPF in Transform Domain)
    # ======================================================
    print("=" * 50)
    print(
        f"{BColors.WARNING}{BColors.BOLD}"
        f"Centering The Ideal Low Pass Filter in the Spatial Domain (LPF in Transform Domain)......"
        f"{BColors.ENDC}{BColors.ENDC}")
    center_lpf_transform_domain = ImageUtils.center_domain(
        image_array=lpf_transform_domain
    )
    print("\nSpatial Centered Ideal Low Pass Filter Array (Transform Domain)\n", center_lpf_transform_domain)

    print(
        f"{BColors.OkGREEN}{BColors.BOLD}"
        f"\nCentering The Ideal Low Pass Filter in the Spatial Domain (LPF in Transform Domain) Finished."
        f"{BColors.ENDC}{BColors.ENDC}")

    # ======================================================
    # Calculating IDFT of The Ideal Low Pass Filter
    # ======================================================
    print("=" * 50)
    print(
        f"{BColors.WARNING}{BColors.BOLD}"
        f"Calculating IDFT of The Ideal Low Pass Filter....."
        f"{BColors.ENDC}{BColors.ENDC}")

    lpf_idft = np.real(
        np.fft.ifft2(center_lpf_transform_domain)
    )

    center_lpf_idft = ImageUtils.center_domain(
        image_array=lpf_idft
    )

    ImageUtils.save_image(
        image_array=ImageUtils.normalize_image(center_lpf_idft),
        filepath=os.path.join(base_dir, f"lpf(r={radius})_idft.png")
    )

    print(
        f"{BColors.OkGREEN}{BColors.BOLD}"
        f"\nCalculating IDFT of The Ideal Low Pass Filter Finished."
        f"{BColors.ENDC}{BColors.ENDC}")

    # ======================================================
    # Creating Visualization
    # ======================================================
    print("=" * 50)
    print(
        f"{BColors.WARNING}{BColors.BOLD}"
        f"Creating Visualization...."
        f"{BColors.ENDC}{BColors.ENDC}")
    images = [
        lpf_transform_domain,           # Original LPF (transform domain)
        center_lpf_idft                 # IDFT of LPF (centered)
    ]
    titles = [
        f"Ideal LPF (Transform Domain) (R={radius})",
        f"IDFT of LPF (Centered)"
    ]
    Visualizer.display_images(images, titles, cols=2, cmap='gray', fig_size=(11, 5),
                              filepath=os.path.join(base_dir, f"image_visualization_(r={radius}).tiff"))

    print(
        f"{BColors.OkGREEN}{BColors.BOLD}"
        f"\nCreating Visualization Finished."
        f"{BColors.ENDC}{BColors.ENDC}")

if __name__ == '__main__':
    main()