from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog


def calculate_moments(image_path):
    image = Image.open(image_path).convert("L")  # Convert the image to grayscale
    width, height = image.size
    moments = []

    for x in range(width):
        for y in range(height):
            pixel_value = image.getpixel((x, y))
            # Assuming white pixels have intensity 255 in a grayscale image
            if pixel_value == 255:
                moment = {
                    "Общая масса": 1,
                    "Момент по X": x,
                    "Момент по Y": y,
                    "Второй момент по X": x**2,
                    "Второй момент по XY": x*y,
                    "Второй момент по Y": y**2,
                    "Третий момент по X": x**3,
                    "Третий момент по XY": x**2*y,
                    "Третий момент по Y": x*y**2,
                    "Третий момент по Y": y**3,
                }
                moments.append(moment)

    return moments


def browse_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
    if file_path:
        moments = calculate_moments(file_path)
        display_moments(moments)


def display_moments(moments):
    # Clear the console before displaying moments
    print("\033c")

    for i, moment in enumerate(moments, 1):
        print(f"Object {i} moments:")
        for key, value in moment.items():
            print(f"{key}: {value}")
        print()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Moments Calculator")

    browse_button = tk.Button(root, text="Browse Image", command=browse_image)
    browse_button.pack(pady=10)

    root.mainloop()
