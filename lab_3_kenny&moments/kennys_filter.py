import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

from scipy.signal import convolve2d


def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image


def show_images(original, processed, title1="Original", title2="Processed"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)

    plt.subplot(1, 2, 2)
    plt.imshow(processed, cmap='gray')
    plt.title(title2)

    plt.show()


def gaussian_blur(image, kernel_size):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * (kernel_size / 2) ** 2)) *
                     np.exp(-((x - (kernel_size - 1) / 2) ** 2 + (y - (kernel_size - 1) / 2) ** 2) / (
                                 2 * (kernel_size / 2) ** 2)),
        (kernel_size, kernel_size)
    )
    kernel /= np.sum(kernel)

    return convolve2d(image, kernel)


def sobel_filter(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gradient_x = convolve2d(image, kernel_x)
    gradient_y = convolve2d(image, kernel_y)

    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    gradient_direction = np.arctan2(gradient_y, gradient_x)

    return gradient_magnitude, gradient_direction


def non_maximum_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    result = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]

            # Quantize the angle to one of four directions
            angle = np.rad2deg(angle)
            angle = (angle + 45) // 90 * 90
            angle = np.deg2rad(angle)

            # Compare with neighbors
            if (angle == 0 and
                    gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1] and
                    gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]):
                result[i, j] = gradient_magnitude[i, j]
            elif (angle == np.pi / 2 and
                  gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j] and
                  gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]):
                result[i, j] = gradient_magnitude[i, j]
            elif (angle == np.pi and
                  gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1] and
                  gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]):
                result[i, j] = gradient_magnitude[i, j]
            elif (angle == 3 * np.pi / 2 and
                  gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j] and
                  gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]):
                result[i, j] = gradient_magnitude[i, j]

    return result


def hysteresis_thresholding(image, low_threshold, high_threshold):
    strong_edges = (image > high_threshold)
    weak_edges = (image >= low_threshold) & (image <= high_threshold)

    result = np.zeros_like(image)

    result[strong_edges] = 100
    result[weak_edges] = 400

    return result


def apply_canny(image, low_threshold, high_threshold, gaussian_kernel_size=10):
    blurred_image = gaussian_blur(image, gaussian_kernel_size)
    gradient_magnitude, gradient_direction = sobel_filter(blurred_image)
    non_max_suppressed = non_maximum_suppression(gradient_magnitude, gradient_direction)
    result = hysteresis_thresholding(non_max_suppressed, low_threshold, high_threshold)

    return result


if __name__ == "__main__":
    # Открытие окна выбора файла
    root = Tk()
    root.withdraw()  # Не отображать основное окно Tkinter
    file_path = filedialog.askopenfilename(title="Выберите изображение",
                                           filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    root.destroy()  # Закрыть Tkinter после выбора файла

    if not file_path:
        print("Выбор файла отменён.")
        exit()

    original_image = read_image(file_path)

    # Задайте значения для пороговых значений нижнего и верхнего порога
    low_threshold = 20
    high_threshold = 10

    # Применение фильтра Кенни
    edges = apply_canny(original_image, low_threshold, high_threshold)

    # Вывод изображений
    show_images(original_image, edges, title1="Original Image", title2="Canny Edges")
