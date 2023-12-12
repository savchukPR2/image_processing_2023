import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# Выберите файл изображения
root = Tk()
root.withdraw()  # скрыть основное окно

file_path = filedialog.askopenfilename(title="Select an Image File",
                                        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

# Загрузите изображение
img = plt.imread(file_path)

# Преобразование в оттенки серого
gray_img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]

# Прямое преобразование Фурье без сторонних библиотек
fft_result = np.fft.fft2(gray_img)
fft_shifted = np.fft.fftshift(fft_result)

# Логарифмическое преобразование для визуализации
log_magnitude_spectrum = np.log(1 + np.abs(fft_shifted))

# Визуализация результатов
plt.figure(figsize=(10, 8))

plt.subplot(231), plt.imshow(gray_img, cmap='gray'), plt.title('Original Image')
plt.subplot(232), plt.imshow(log_magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')

# Запрос пользователя о месте сохранения Magnitude Spectrum в формате .npy
output_file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("NumPy files", "*.npy")])

# Сохранение Magnitude Spectrum в указанное место в формате .npy
if output_file_path:
    np.save(output_file_path, log_magnitude_spectrum)

plt.show()
