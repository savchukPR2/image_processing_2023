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


def process_image(image_path, filter_type, filter_param):
    # Загрузка цветного изображения
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if filter_type == 'median':
        # Применение медианного фильтра к каждому цветовому каналу
        filtered_image = apply_median_filter(original_image, filter_param)
        filter_name = f'Median Filter (Window Size {filter_param})'
    elif filter_type == 'gaussian':
        # Создание Gaussian kernel
        sigma = filter_param
        kernel_size = int(6 * sigma + 1)
        gaussian_kernel_matrix = gaussian_kernel(kernel_size, sigma)

        # Применение Gaussian фильтра
        filtered_image = apply_gaussian_filter(original_image, gaussian_kernel_matrix)
        filter_name = f'Gaussian Filter (Sigma {filter_param})'
    else:
        raise ValueError("Unknown filter type")

    # Отображение оригинального и фильтрованного изображений
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title(filter_name)
    axes[1].axis('off')

    # Вычисление MSE и UIQI
    m = mse(original_image, filtered_image)
    s = calculate_uiqi(original_image, filtered_image)

    # Вывод результатов на графический интерфейс
    result_text = f'MSE: {m:.2f}\nUIQI: {s:.2f}'

    # Создание нового окна Tkinter
    root = Tk()
    root.title("Image Processing Results")

    # Встраивание графика в Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    # Добавление метки с результатами
    result_label = Label(root, text=result_text, font=("Helvetica", 12))
    result_label.pack()

    # Показать только график, который уже встроен в Tkinter
    canvas.draw()

    # Запуск основного цикла Tkinter


def main():
    # Выбор файла изображения
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg")])

    if file_path:
        # Задание типа и параметра фильтра
        filter_type = input("Enter filter type ('median' or 'gaussian'): ")
        filter_param = int(input("Enter filter parameter: "))

        # Обработка изображения
        process_image(file_path, filter_type, filter_param)
    root.mainloop()


if __name__ == "__main__":
    main()
