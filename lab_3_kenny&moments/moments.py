import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def calculate_moments_and_histogram(image_path):
    # Загрузка изображения в оттенках серого
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Вычисление моментов Ху
    moments = cv2.moments(image)

    # Расчет гистограммы
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # Отображение изображения и гистограммы
    plot_image_and_histogram(image, hist)

def plot_image_and_histogram(image, hist):
    # Создание фигуры и осей с явным указанием размеров
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Отображение изображения
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Image')

    # Отображение гистограммы
    ax2.plot(hist, color='black')

    # Установка максимального значения по оси Y
    ax2.set_ylim(0, 4000)

    ax2.set_title('Histogram')

    # Уменьшение размера гистограммы по вертикальной оси
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    # Отображение окна
    plt.show()

def select_image():
    Tk().withdraw()  # чтобы окно выбора файла не отображалось
    file_path = filedialog.askopenfilename(title='Select an image', filetypes=[('Image files', '*.png;*.jpg;*.jpeg;*.bmp;*.gif')])

    if file_path:
        calculate_moments_and_histogram(file_path)

if __name__ == "__main__":
    # Запуск выбора изображения
    select_image()