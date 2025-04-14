import tensorflow as tf
import numpy as np
import cv2
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image

# Model yükle
model = tf.keras.models.load_model('meral_model.h5')

# Class isimlerin
class_names = ['avsar', 'bicak', 'catal', 'defter', 'kalem', 'kasik', 'parfum', 'sirma']

# Dosya seçtir
Tk().withdraw()
image_path = askopenfilename(title="Fotoğraf Seç", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if image_path:
    print("Seçilen Fotoğraf:", image_path)

    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((256, 256))
        img = np.array(img)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = round(100 * np.max(prediction), 2)

        print(f"Tahmin Edilen Nesne: {class_name}")
        print(f"Doğruluk Oranı: %{confidence}")

    except Exception as e:
        print("Fotoğraf açılırken hata oluştu:", e)

else:
    print("Fotoğraf seçilmedi!")
