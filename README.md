# MATLAB ile Derin Öğrenme Tabanlı Araç Modeli Tanıma Sistemi 🚗

Bu proje, MATLAB ve **ResNet-50 (Transfer Learning)** mimarisini kullanarak görüntüden araç marka ve modelini tespit eden, yüksek başarımlı bir yapay zeka sistemidir.

## 🎯 Proje Hakkında
Bu çalışmada, Türkiye yollarında sıkça karşılaşılan ve birbirine benzeyen araç modelleri sınıflandırılmıştır. Başlangıçta AlexNet ile yapılan denemeler (%84.5) yeterli görülmemiş, **ResNet-50** mimarisine geçilerek başarı oranı **%95.27** seviyesine çıkarılmıştır.

* **Yöntem:** Transfer Learning (ResNet-50 Mimarisi)
* **Doğruluk Oranı (Accuracy):** %95.27 🏆
* **Veri Seti:** 7 Farklı Sınıf (Duster, Corolla, Şahin, Civic, Palio vb.)
* **Platform:** MATLAB 2023b + Deep Learning Toolbox + GPU (RTX 3060)

## 📂 Proje İçeriği
* `src/`: Veri işleme, eğitim (ResNet-50) ve test kodları.
* `data/`: Etiketlenmiş veri setine ait .mat dosyaları (Ground Truth).
* `models/`: Eğitilmiş final modeli (Final_Model_ResNet50.mat).
* `results/`: Eğitim grafikleri ve Confusion Matrix analizleri.

## 📊 Sonuçlar

**Final Başarı Tablosu (Confusion Matrix):**
![Confusion Matrix](results/Confusion_Matrix_Final_95.png)

## 🛠️ Kullanılan Teknolojiler
* MATLAB & Image Labeler App
* **ResNet-50** Pre-trained Network
* NVIDIA RTX 3060 GPU Computing