import cv2
import numpy as np
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
import os

def add_exponential_noise(image, scale):
    # Генерация экспоненциального шума
    noise = np.random.exponential(scale, image.shape)

    # Добавление шума к изображению
    noisy_image = image + noise

    # Ограничение значений пикселей от 0 до 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    return noisy_image, noise

def exponential_noise_pixel_histogram(noise):
    # Построение гистограммы пикселей экспоненциального шума
    plt.figure(figsize=(8, 4))
    plt.hist(noise.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title('Exponential Noise Pixel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def process_image(image_path, noise_scale, save_directory, save_filename):
    # Загрузка изображения
    original_image = cv2.imread(image_path)

    # Преобразование изображения из BGR в RGB
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Генерация экспоненциального шума
    noise = np.random.exponential(noise_scale, original_image.shape)

    # Добавление шума к изображению
    noisy_image = original_image_rgb + noise

    # Ограничение значений пикселей от 0 до 255
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

    # Отображение оригинального и зашумленного изображений
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image_rgb)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(noisy_image)
    plt.title('Noisy Image with Exponential Noise')
    plt.axis('off')

    plt.show()

    # Сохранение изображения с шумом в выбранную директорию с выбранным именем
    save_path = os.path.join(save_directory, save_filename)
    cv2.imwrite(save_path, cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR))
    print(f'Noisy image saved at: {save_path}')

    # Отображение гистограммы пикселей экспоненциального шума
    exponential_noise_pixel_histogram(noise)

def main():
    # Выбор файла изображения
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg;*.jpeg")])

    if file_path:
        # Задание масштаба экспоненциального шума
        noise_scale = 50  # Можно изменить по необходимости

        # Выбор директории для сохранения
        save_directory = filedialog.askdirectory(title="Select Directory to Save Image")

        if save_directory:
            # Ввод желаемого имени файла
            save_filename = filedialog.asksaveasfilename(
                title="Save Image As",
                initialdir=save_directory,
                filetypes=[("JPEG files", "*.jpg")],
                defaultextension=".jpg"
            )

            if save_filename:
                # Обработка изображения
                process_image(file_path, noise_scale, save_directory, save_filename)

if __name__ == "__main__":
    main()