import numpy as np
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
            -((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

def compute_gradients(image):
    # Градиенты изображения по оси X и Y
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисление амплитуды градиента
    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Вычисление угла градиента
    gradient_direction = np.arctan2(sobel_y, sobel_x)

    return gradient_magnitude, gradient_direction

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    rows, cols = gradient_magnitude.shape
    suppressed = np.zeros_like(gradient_magnitude)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_direction[i, j]
            mag = gradient_magnitude[i, j]

            x, y = 0, 0

            # Определение направления градиента
            if (0 <= angle < np.pi / 4) or (7 * np.pi / 4 <= angle <= 2 * np.pi):
                x, y = 1, 0
            elif (np.pi / 4 <= angle < 3 * np.pi / 4) or (5 * np.pi / 4 <= angle < 7 * np.pi / 4):
                x, y = 0, 1
            elif (3 * np.pi / 4 <= angle < 5 * np.pi / 4) or (9 * np.pi / 4 <= angle <= 2 * np.pi):
                x, y = -1, 0

            # Подавление немаксимумов
            if mag >= gradient_magnitude[i + x, j + y] and mag >= gradient_magnitude[i - x, j - y]:
                suppressed[i, j] = mag

    return suppressed

def threshold(image, low_threshold, high_threshold):
    high_value = 255
    low_value = 50  # Измените этот порог по необходимости

    strong_edges = (image >= high_threshold)
    weak_edges = (image >= low_threshold) & (image < high_threshold)

    # Присвоение значений пикселям в зависимости от порогов
    image[strong_edges] = high_value
    image[weak_edges] = low_value

    return image

def hysteresis(image, weak_value, strong_value):
    rows, cols = image.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if image[i, j] == weak_value:
                # Проверка вокруг пикселя
                if (image[i - 1:i + 2, j - 1:j + 2] == strong_value).any():
                    image[i, j] = strong_value
                else:
                    image[i, j] = 0

    return image

def canny_edge_detection(image, low_threshold, high_threshold):
    # Set the value of sigma for the Gaussian filter
    sigma = 1.5
    kernel_size = int(6 * sigma + 1)

    # Create Gaussian kernel
    gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)

    # Apply Gaussian filter
    blurred_image = apply_gaussian_filter(image, gaussian_kernel_matrix)

    # Compute gradients
    gradient_magnitude, gradient_direction = compute_gradients(blurred_image)

    # Non-maximum suppression
    suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)

    # Thresholding
    thresholded_image = threshold(suppressed_image, low_threshold, high_threshold)

    # Hysteresis
    final_image = hysteresis(thresholded_image, 50, 255)  # Adjust these values accordingly

    return final_image

def plot_canny_result(original, edges, title1, title2):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title(title1)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(title2)
    plt.axis('off')

    plt.show()

def main():
    # Create Tk instance
    root = Tk()
    root.withdraw()

    # Select an image from the computer
    image_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg")])

    if image_path:
        # Load the image
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Set the values for Canny edge detection
        low_threshold = 50
        high_threshold = 75

        # Set the value of sigma for the Gaussian filter
        sigma = 20
        kernel_size = int(6 * sigma + 1)
        gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)

        # Apply Canny edge detection
        edges = canny_edge_detection(original_image, low_threshold, high_threshold)

        # Visualize the results
        plot_canny_result(original_image, edges, 'Original Image', 'Canny Edges')

    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()