import tensorflow as tf
import matplotlib.pyplot as plt

# Modeli yükle
model = tf.keras.models.load_model('meral_model_yeni.h5')

# Test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    'test_tf',
    image_size=(256, 256),
    batch_size=32)

# Test et
loss, accuracy = model.evaluate(test_ds)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Grafik
labels = ['Loss', 'Accuracy']
values = [loss, accuracy]

plt.figure(figsize=(5, 5))
plt.bar(labels, values, color=['red', 'green'])
plt.title('Test Sonuçları')
plt.ylim(0, 1)
plt.ylabel('Değer')
plt.show()
