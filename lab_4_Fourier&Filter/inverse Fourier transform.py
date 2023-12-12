import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from tkinter import Tk

# Выберите файл изображения
root = Tk()
root.withdraw()  # скрыть основное окно

# file_path = filedialog.askopenfilename(title="Select an Image File",
#                                         filetypes=[("NumPy files", "*.npy")])
file_path = 'cats_magnitude.npy'

# Загрузите результат обратного преобразования Фурье из файла
ifft_result = np.load(file_path)

# Визуализация результата
plt.imshow(ifft_result, cmap='gray')
plt.title('Reconstructed Image')
plt.show()
