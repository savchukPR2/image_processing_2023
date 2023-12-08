import cv2
from tkinter import Tk, filedialog, Label
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.ndimage import median_filter
import numpy as np


def gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def apply_gaussian_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


def apply_median_filter(image, window_size):
    filtered_channels = [median_filter(channel, size=(window_size, window_size)) for channel in cv2.split(image)]
    return cv2.merge(filtered_channels)


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def calculate_uiqi(image1, image2):
    image1 = image1.astype(float) / 255.0
    image2 = image2.astype(float) / 255.0

    mean1 = np.mean(image1)
    mean2 = np.mean(image2)

    var1 = np.var(image1)
    var2 = np.var(image2)

    covar = np.mean((image1 - mean1) * (image2 - mean2))

    num = 4 * covar * mean1 * mean2
    den = (var1 + var2) * (mean1 ** 2 + mean2 ** 2)

    uiqi = num / den

    return uiqi


def load_noisy_image(image_path):
    # Загрузка цветного изображения с шумом
    noisy_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    return noisy_image

def main():
    # Выбор файла изображения
    root = Tk()
    root.withdraw()
    original_file_path = filedialog.askopenfilename(title="Select Original Image", filetypes=[("Image files", "*.jpg;*.jpeg")])
    noisy_file_path = filedialog.askopenfilename(title="Select Noisy Image", filetypes=[("Image files", "*.jpg;*.jpeg")])

    if original_file_path and noisy_file_path:
        # Загрузка оригинального изображения и изображения с шумом
        original_image = cv2.imread(original_file_path, cv2.IMREAD_COLOR)
        noisy_image = load_noisy_image(noisy_file_path)

        # Задание типа и параметра фильтра
        filter_type = input("Enter filter type ('median' or 'gaussian'): ")
        filter_param = int(input("Enter filter parameter: "))

        # Применение фильтра к изображению с шумом
        if filter_type == 'median':
            filtered_image = apply_median_filter(noisy_image, filter_param)
        elif filter_type == 'gaussian':
            sigma = filter_param
            kernel_size = int(6 * sigma + 1)
            gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)
            filtered_image = apply_gaussian_filter(noisy_image, gaussian_kernel_matrix)
        else:
            raise ValueError("Unknown filter type")

        # Вычисление MSE и UIQI между оригинальным и фильтрованным изображением
        mse_value = mse(original_image, filtered_image)
        uiqi_value = calculate_uiqi(original_image, filtered_image)

        # Отображение результатов
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Noisy Image')
        axes[1].axis('off')
        axes[2].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Filtered Image ({filter_type} Filter)')
        axes[2].axis('off')

        result_text = f'MSE: {mse_value:.2f}\nUIQI: {uiqi_value:.2f}'

        root = Tk()
        root.title("Image Processing Results")

        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.get_tk_widget().pack()

        result_label = Label(root, text=result_text, font=("Helvetica", 12))
        result_label.pack()

        canvas.draw()
        root.mainloop()

if __name__ == "__main__":
    main()
