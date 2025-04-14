import tensorflow as tf
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from PIL import Image
import cv2

model = tf.keras.models.load_model('meral_model_yeni.h5')

class_names = ['avsar', 'bicak', 'catal', 'defter', 'kalem', 'kasik', 'parfum', 'sirma']

Tk().withdraw()
image_path = askopenfilename(title="Fotoğraf Seç", filetypes=[("Image files", "*.jpg *.jpeg *.png")])

if image_path:
    print("Seçilen Fotoğraf:", image_path)

    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((256, 256))
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        class_name = class_names[class_index]
        confidence = round(100 * np.max(prediction), 2)

        print(f"Tahmin Edilen Nesne: {class_name}")
        print(f"Doğruluk Oranı: %{confidence}")

        # Görüntüyü PIL'den cv2'ye çevir
        img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Üzerine tahmin yaz
        cv2.putText(img_cv2, f'{class_name} - %{confidence}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Tahmin Sonucu", img_cv2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Fotoğraf açılırken hata oluştu:", e)

else:
    print("Fotoğraf seçilmedi!")
