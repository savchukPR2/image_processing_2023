import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk

class FourierTransformApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fourier Transform")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.calculate_button = tk.Button(root, text="Perform Fourier Transform", command=self.perform_fourier_transform)
        self.calculate_button.pack()

        self.result_label = tk.Label(root, text="Results will be displayed here.")
        self.result_label.pack()

        # Устанавливаем размер окна
        self.root.geometry("400x400")

    def perform_fourier_transform(self):
        file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

        if file_path:
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            # Perform Fourier Transform
            f_transform = self.custom_fft(image)

            # Display the original and transformed images
            self.display_image(image, "Original Image")
            self.display_results(f_transform, "Fourier Transform")

    def custom_fft(self, image):
        # Perform 1D FFT along rows
        f_transform_rows = np.fft.fft(image, axis=1)

        # Perform 1D FFT along columns
        f_transform = np.fft.fft(f_transform_rows, axis=0)

        return f_transform

    def display_image(self, image, title):
        img_tk = ImageTk.PhotoImage(Image.fromarray(image))
        self.image_label.config(image=img_tk)
        self.image_label.image = img_tk

        # Update the result label
        self.result_label.config(text=f"{title} displayed.")

    def display_results(self, data, title):
        # For simplicity, display only a portion of the transformed data
        display_data = np.log(np.abs(data[:10, :10]) + 1)

        self.result_label.config(text=f"{title}:\n{display_data}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FourierTransformApp(root)
    root.mainloop()
