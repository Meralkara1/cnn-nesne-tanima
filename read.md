# CNN ile Nesne Tanıma Projesi - Meral Kara

## Proje Hakkında
Bu projede kendi çektiğim görsellerle 8 farklı nesneyi tanıyabilen bir derin öğrenme modeli geliştirdim. Model, CNN (Convolutional Neural Network) mimarisi ile oluşturulmuş ve TensorFlow/Keras kullanılarak eğitilmiştir.

## Kullanılan Nesne Sınıfları
- avsar  
- bicak  
- catal  
- defter  
- kalem  
- kasik  
- parfum  
- sirma  

## Kullanılan Teknolojiler
- Python  
- Tensorflow / Keras  
- Matplotlib  
- OpenCV  
- PIL  

## Model Mimarisi
- Data Augmentation (Veri Çoğaltma)  
- 3 adet Conv2D Katmanı  
- MaxPooling  
- Dropout  
- Dense Katmanları  
- Softmax Çıkış Katmanı  

## Karşılaştığım Problemler ve Çözümler

| Problem | Sebep | Çözüm |
|---------|-------|-------|
|Model sürekli "sirma" tahmin ediyordu|Overfitting (Ezberleme)|Epoch sayısı azaltıldı, Dropout ve Data Augmentation eklendi, model yeniden eğitildi.|
|Kamera ile yanlış tahmin problemi|class_names sırası hatalıydı|class_names güncellendi.|
|Fotoğraf yüklerken hata|cv2.imread bazı fotoğrafları okuyamıyordu|PIL ile okuma ve cv2'ye dönüştürme yaparak çözüldü.|

## Eğitim Sonuçları Grafik

![Loss Grafiği](loss.png)

![Accuracy Grafiği](accuracy.png)

## Test Sonuçları

| Metrik | Değer |
|--------|-------|
|Loss|1.15|
|Accuracy|%74.49|

![Test Sonucu](test2_sonuc.png)

## Fotoğraf ile Tahmin Örneği

![Fotoğraf Tahmin](foto_tahmin.png)

## Kamera ile Canlı Tahmin Örneği

![Kamera Tahmin](kamera_tahmin.png)

## Dosya Yapısı

```
egitim1.py                → İlk model denemesi (hatalı)  
egitim2.py                → Final model eğitimi  
test2.py                  → Model test ve değerlendirme  
kamera.py                 → Kamera ile canlı deneme  
foto_tahmin.py            → Fotoğraf seçerek deneme  
meral_model_yeni.h5       → Final model dosyası  
loss.png                  → Loss grafiği  
accuracy.png              → Accuracy grafiği  
test2_sonuc.png           → Test sonuç görseli  
foto_tahmin.png           → Fotoğraf tahmin görseli  
kamera_tahmin.png         → Kamera tahmin görseli  
train_tf/                 → Eğitim verisi  
valid_tf/                 → Doğrulama verisi  
test_tf/                  → Test verisi  
```

## Projeyi Çalıştırmak İçin

Model Eğitimi:
```python
python egitim2.py
```

Model Test:
```python
python test2.py
```

Kamera ile Deneme:
```python
python kamera.py
```

Fotoğraf ile Deneme:
```python
python foto_tahmin.py
```

## Teşekkürler
Bu projede hem model geliştirdim hem de karşılaştığım problemleri çözerek kendimi geliştirme fırsatı buldum.
