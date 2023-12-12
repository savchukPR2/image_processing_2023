from PIL import Image
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt

def threshold_binary(image, threshold):
    gray_image = image.convert("L")
    pixel_data = list(gray_image.getdata())
    binary_data = [0 if pixel < threshold else 255 for pixel in pixel_data]
    binary_image = Image.new("L", gray_image.size)
    binary_image.putdata(binary_data)
    return binary_image

def main():
    # Создаем корневое окно tkinter (скрытое)
    root = Tk()
    root.withdraw()

    # Запрос пользователя на выбор изображения
    file_path = filedialog.askopenfilename(title="Select an Image File",
                                            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])

    # Проверка, что файл был выбран
    if not file_path:
        print("Выбор файла отменен.")
        return

    # Загрузка изображения
    original_image = Image.open(file_path)

    # Закрытие корневого окна tkinter
    root.destroy()

    # Установка порога для бинаризации
    threshold_value = 100

    # Применение пороговой бинаризации
    binary_result = threshold_binary(original_image, threshold_value)

    # Визуализация результатов на одном экране
    plt.figure(figsize=(8, 4))

    plt.subplot(121)
    plt.imshow(original_image)
    plt.title('Original Image')

    plt.subplot(122)
    plt.imshow(binary_result)
    plt.title('Binary Thresholding Result')

    plt.show()

if __name__ == "__main__":
    main()
