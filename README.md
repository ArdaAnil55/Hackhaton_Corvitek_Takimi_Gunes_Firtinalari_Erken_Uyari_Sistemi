# Hackhaton_Gunes_Firtinalari_Erken_Uyari_Sistemi

# 🌞 Solar Sentinel: Real-Time Space Weather & AI Early Warning System

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)
![XGBoost](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange.svg)
![Gemini AI](https://img.shields.io/badge/AI%20Assistant-Google%20Gemini-purple.svg)
![Three.js](https://img.shields.io/badge/3D%20Globe-Three.js-black.svg)

**Solar Sentinel**, NASA ve NOAA verilerini gerçek zamanlı olarak işleyerek Dünya'yı etkileyebilecek jeomanyetik güneş fırtınalarını (Kp İndeksi) tahmin eden yapay zeka destekli bir erken uyarı ağıdır.

## 🚀 Temel Özellikler ve Mimari

Sistem üç ana teknolojik sütun üzerine inşa edilmiştir:

* **📡Veri Akışı:** 
NASA DONKI ve NOAA SWPC servislerinden gelen anlık Güneş rüzgarı (hız, yoğunluk) ve manyetik alan (Bz, Bt) verileri `asyncio` ve `httpx` kullanılarak darboğaz yaratmadan çekilir. 24 saatlik kayan önbellekte (rolling cache) işlenerek anında analize hazır hale getirilir.
* **🧠"Delta Kp" Tahmini (XGBoost):**
  10 yıllık NASA geçmiş verisiyle eğitilen makine öğrenmesi modelimiz, basit tahminler yerine **"Mevcut durum ne kadar değişecek?" (Delta Kp)** sorusuna odaklanır. Dinamik basınç ve VBs gibi 62 farklı özelliği analiz ederek 3 saat sonrasının fırtına riskini yüksek doğrulukla hesaplar.
* **🌞Sunny AI:**
  Sisteme entegre edilen "Sunny" adlı yapay zeka asistanı, doğrudan istemci (browser) erişimine kapalıdır. Flask tabanlı bir **Chat Proxy** arkasında çalışarak API anahtarı sızıntılarını engeller. Ayrıca, aşırı yüklenmelere karşı **"Quota Defender"** (20 Saniyelik Hard Cooldown) mekanizması ile API kotalarını optimize eder ve sistemin çökmesini önler.

## 🛠️ Kullanılan Teknolojiler
* **Backend:** Python, Flask, Asyncio, Pandas, NumPy
* **Makine Öğrenmesi:** XGBoost, Scikit-Learn
* **Frontend:** HTML5, Tailwind CSS, JavaScript, Three.js
* **Yapay Zeka:** Google Gemini 1.5 Flash API
* **Veri Kaynakları:** NOAA Space Weather, NASA OMNI & DONKI
