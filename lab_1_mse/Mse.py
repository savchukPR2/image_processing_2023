import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2


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


def select_image(label, image_var):
    file_path = filedialog.askopenfilename(title="Select Image")
    if file_path:
        image_var.set(file_path)
        label.config(text=file_path)


def compare_images():
    # Get file paths from StringVar
    image_path1 = image_var1.get()
    image_path2 = image_var2.get()

    # Load images using OpenCV
    image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # Check if images are loaded successfully
    if image1 is not None and image2 is not None:
        # Resize images to a common size
        target_size = (400, 400)
        image1 = cv2.resize(image1, target_size)
        image2 = cv2.resize(image2, target_size)

        # Compare images
        m = mse(image1, image2)
        s = ssim(image1, image2)

        # Display the result
        result_text = f'MSE: {m:.2f}\nUIQI: {s:.2f}'
        result_label.config(text=result_text)
    else:
        result_label.config(text="Error loading images. Check file paths.")


def load_images():
    compare_images(image_var1.get(), image_var2.get())


# Create main window
root = tk.Tk()
root.title("Image Comparison App")

# Image variables
image_var1 = tk.StringVar()
image_var2 = tk.StringVar()

# UI components
label1 = tk.Label(root, text="Image 1:")
label2 = tk.Label(root, text="Image 2:")
result_label = tk.Label(root, text="Comparison results will appear here.")

select_button1 = tk.Button(root, text="Select Image 1", command=lambda: select_image(label1, image_var1))
select_button2 = tk.Button(root, text="Select Image 2", command=lambda: select_image(label2, image_var2))
compare_button = tk.Button(root, text="Compare Images", command=compare_images)


# Arrange components on the grid
label1.grid(row=0, column=0, padx=10, pady=10)
label2.grid(row=1, column=0, padx=10, pady=10)
result_label.grid(row=3, column=0, columnspan=2, pady=10)

select_button1.grid(row=0, column=1, padx=10, pady=10)
select_button2.grid(row=1, column=1, padx=10, pady=10)
compare_button.grid(row=2, column=0, columnspan=2, pady=10)

# Start the GUI event loop
root.mainloop()
