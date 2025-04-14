import tensorflow as tf
import matplotlib.pyplot as plt

# Modeli yükle
model = tf.keras.models.load_model('meral_model.h5')

# Test verisini yükle
test_ds = tf.keras.utils.image_dataset_from_directory(
    'test_tf',
    image_size=(256, 256),
    batch_size=32)

# Modeli test et
loss, accuracy = model.evaluate(test_ds)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Grafik için değerler
labels = ['Loss', 'Accuracy']
values = [loss, accuracy]

# Grafik çizimi
plt.figure(figsize=(5, 5))
plt.bar(labels, values, color=['red', 'green'])
plt.title('Test Sonuçları')
plt.ylim(0, 1)  # 0 ile 1 arası göster
plt.ylabel('Değer')
plt.show()
