import os
import shutil

# Senin sınıf isimlerin sırayla buraya gelecek
class_names = ['catal', 'bicak', 'kasik', 'avsar', 'sirma', 'kalem', 'defter', 'parfum']

# Hangi klasörü ayıracağımız (train, valid, test)
base_path = "test"  # burayı valid veya test yaparak 3 kez çalıştıracağız
images_path = os.path.join(base_path, "images")
labels_path = os.path.join(base_path, "labels")

# Çıkacak olan TensorFlow klasör yapısı
output_path = f"{base_path}_tf"
os.makedirs(output_path, exist_ok=True)

# Sınıf klasörlerini oluştur
for cname in class_names:
    os.makedirs(os.path.join(output_path, cname), exist_ok=True)

# Label dosyalarını oku ve uygun klasöre taşı
for label_file in os.listdir(labels_path):
    label_path = os.path.join(labels_path, label_file)

    with open(label_path, "r") as file:
        lines = file.readlines()

    if not lines:
        continue  # boş label varsa geç

    first_line = lines[0].strip().split(" ")[0]  # ilk satırdaki sınıf indexi
    class_index = int(first_line)
    class_name = class_names[class_index]

    # Fotoğrafın adını bul
    image_filename = label_file.replace(".txt", ".jpg")
    src_image_path = os.path.join(images_path, image_filename)
    dst_image_path = os.path.join(output_path, class_name, image_filename)

    if os.path.exists(src_image_path):
        shutil.copy(src_image_path, dst_image_path)
