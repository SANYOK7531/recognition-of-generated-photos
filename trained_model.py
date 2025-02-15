import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
loaded_model_1 = load_model("C:/Users/Oleksandr/Desktop/Universaty/NN/Models/trained_model_1.h5")
loaded_model_2 = load_model("C:/Users/Oleksandr/Desktop/Universaty/NN/Models/trained_model_2.h5")
loaded_model_3 = load_model("C:/Users/Oleksandr/Desktop/Universaty/NN/Models/trained_model_3.h5")

current_image_path = None

def predict_image(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Image not loaded: {image_path}")
        return "Image not loaded"
    img = cv2.resize(img, (100, 100))
    img = img / 255.0

    # Add an extra dimension for the batch size
    img = np.expand_dims(img, axis=0)

    predict = []
    predict_percent = []

    prediction_1 = loaded_model_1.predict(img)
    prediction_1_percent = round(prediction_1[0][0] * 100, 2)
    if prediction_1 < 0.5:
        predict.append("Real")
        predict_percent.append(f" ({100-prediction_1_percent}%)")
    else:
        predict.append("Generated")
        predict_percent.append(f" ({prediction_1_percent}%)")

    prediction_2 = loaded_model_2.predict(img)
    prediction_2_percent = round(prediction_2[0][0] * 100, 2)
    if prediction_2 < 0.5:
        predict.append("Real")
        predict_percent.append(f" ({100-prediction_2_percent}%)")
    else:
        predict.append("Generated")
        predict_percent.append(f" ({prediction_2_percent}%)")

    prediction_3 = loaded_model_3.predict(img)
    prediction_3_percent = round(prediction_3[0][0] * 100, 2)
    if prediction_3 < 0.5:
        predict.append("Real")
        predict_percent.append(f" ({100-prediction_3_percent}%)")
    else:
        predict.append("Generated")
        predict_percent.append(f" ({prediction_3_percent}%)")

    return predict, predict_percent

def most_common(lst):
    return max(set(lst), key=lst.count)

def open_image():
    global current_image_path
    current_image_path = filedialog.askopenfilename()
    if current_image_path:
        image = Image.open(current_image_path)
        image.thumbnail((300, 300))  # Resize image to fit within 300x300
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo  # Keep a reference to avoid garbage collection
        # Clear previous prediction result
        prediction_label_1.config(text="Prediction 1: ")
        prediction_label_2.config(text="Prediction 2: ")
        prediction_label_3.config(text="Prediction 3: ")
        prediction_label_4.config(text="Prediction final: ")


def predict_button_clicked():
    # Get the selected image path
    if current_image_path:
        # Predict the loaded image
        prediction_result, predict_percent = predict_image(current_image_path)
        prediction_label_1.config(text="Prediction 1: " + prediction_result[0] + predict_percent[0])
        prediction_label_2.config(text="Prediction 2: " + prediction_result[1] + predict_percent[1])
        prediction_label_3.config(text="Prediction 3: " + prediction_result[2] + predict_percent[2])
        prediction_label_4.config(text="Prediction final: " + most_common(prediction_result))

# Create the main window
root = tk.Tk()
root.title("Image Predictor")

# Set the base window size (width x height)
root.geometry("500x700")

# Make the window not resizable
root.resizable(False, False)

# Set background color
root.configure(bg="#4F4C4B")

# Create a button to open an image
open_button = tk.Button(root, text="Open Image", command=open_image, bg="#4F4C4B", fg="white", font=("Arial", 12))
open_button.pack(pady=10)

# Create a label to display the image with the specified background color
image_label = tk.Label(root, bg="#4F4C4B")
image_label.pack()

# Create a button to predict
predict_button = tk.Button(root, text="Predict", command=predict_button_clicked, bg="#4F4C4B", fg="white", font=("Arial", 12))
predict_button.pack(pady=10)

# Create a label to display the prediction result
prediction_label_1 = tk.Label(root, text="Prediction 1: ",  bg="#4F4C4B", fg="white", font=("Arial", 12))
prediction_label_2 = tk.Label(root, text="Prediction 2: ",  bg="#4F4C4B", fg="white", font=("Arial", 12))
prediction_label_3 = tk.Label(root, text="Prediction 3: ",  bg="#4F4C4B", fg="white", font=("Arial", 12))
prediction_label_4 = tk.Label(root, text="Prediction final: ",  bg="#4F4C4B", fg="white", font=("Arial", 12))
prediction_label_1.pack()
prediction_label_2.pack()
prediction_label_3.pack()
prediction_label_4.pack()
# Start the main event loop
root.mainloop()
