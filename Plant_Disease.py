import numpy as np
import os
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Eğitilmiş modeli yükle
model = load_model("plant_disease_model.h5")
print("Kaydedilmiş model başarıyla yüklendi!")

# Test klasöründeki tüm dosyaları kontrol et
test_folder_path = 'C:/Users/grkna/Downloads/Test'
all_files = os.listdir(test_folder_path)

# Görüntü boyutları
IMG_WIDTH, IMG_HEIGHT = 128, 128

# Her bir test görüntüsünü işle ve tahmini yap
def predict_image(img_path, model, class_names):
    # Görüntüyü yükleme ve ön işleme
    img = load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(img) / 255.0  # Normalizasyon
    img_array = np.expand_dims(img_array, axis=0)  # Model için boyut ekleme

    # Tahmin yapma
    prediction = model.predict(img_array)
    predicted_class_idx = np.argmax(prediction)  # En yüksek olasılık
    predicted_class = class_names[predicted_class_idx]  # Sınıf adı
    confidence = prediction[0][predicted_class_idx]  # Tahmin doğruluk oranı (olasılık)

    return predicted_class, confidence

# Test veri kümesindeki sınıflar
classes = os.listdir('C:/Users/grkna/Downloads/dataset/')

# Tahmin yap ve sonuçları yazdır
results = []
for file_name in all_files:
    img_path = os.path.join(test_folder_path, file_name)
    predicted_class, confidence = predict_image(img_path, model, classes)
    results.append((file_name, predicted_class, confidence))

# Sonuçları ekrana yazdır
print("Test Sonuçları:")
for file_name, predicted_class, confidence in results:
    print(f"Görüntü: {file_name} - Tahmin Edilen Sınıf: {predicted_class} - Doğruluk Oranı: %{confidence * 100:.2f}")