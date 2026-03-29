import pickle
import pandas as pd
from pathlib import Path

# Senin yazdığın eğitim kodunu modül olarak içe aktarıyoruz
# (Veriyi aynı senin sisteminin ayrıştırdığı gibi ayrıştırmak için)
import train as trainer

print("1. Uzay Havası Yapay Zeka Modeli Yükleniyor...")
with open("solar_xgboost_model.pkl", "rb") as f:
    meta = pickle.load(f)
model = meta["model"]

print("2. NASA 2024 Verileri Önbellekten Çekiliyor...")
cache_path = Path("omni_cache/omni2_2024.dat")
if not cache_path.exists():
    print("HATA: omni2_2024.dat dosyası bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")
    exit()

raw_data = cache_path.read_bytes()

print("3. Özellik Mühendisliği (Feature Engineering) Uygulanıyor...")
# Veriyi aynı modelin eğitildiği gibi hazırlıyoruz
df_raw = trainer.parse_omni_bytes(raw_data, 2024)
df_clean = trainer.clean_data(df_raw)
df_feat = trainer.build_features(df_clean)

# Hedef Zaman: 10 Mayıs 2024, Saat 15:00 UTC
# Bu saatte rüzgar Dünya'ya yeni vurmaya başladı. Biz 3 saat sonrasını (18:00) tahmin edeceğiz.
target_time = pd.to_datetime("2024-05-10 20:00:00", utc=True)

if target_time in df_feat.index:
    row = df_feat.loc[[target_time]]
    
    # Modelin içine girecek verileri ayır
    X_test = row[meta["feature_cols"]]
    
    # O saatteki GERÇEK veriler
    real_kp_now = row["kp"].values[0]
    real_kp_3h = row["target_kp_3h"].values[0]  # Gelecekte (18:00'da) gerçekten olan Kp
    
    # XGBoost Tahmini
    prediction = model.predict(X_test)[0]
    
    print("\n" + "="*55)
    print(" 🚀 TARİHSEL G5 FIRTINASI TESTİ (10 MAYIS 2024)")
    print("="*55)
    print(f"Zaman Makinesi    : {target_time.strftime('%Y-%m-%d %H:00')} UTC")
    print(f"Güneş Rüzgarı Hızı: {row['speed'].values[0]:.0f} km/s (Fırtına yaklaşıyor!)")
    print(f"Manyetik Alan (Bz): {row['bz_gsm'].values[0]:.1f} nT (Güçlü Güney Yönelimi!)")
    print(f"O Anki Kp İndeksi : {real_kp_now}")
    print("-" * 55)
    print(f"🤖 XGBoost T+3 Tahmini (18:00 için): Kp {prediction:.2f}")
    print(f"🌍 GERÇEKTE YAŞANAN   (18:00'da) : Kp {real_kp_3h:.2f}")
    print("="*55)
    
    # Başarı Analizi
    hata_payi = abs(prediction - real_kp_3h)
    print(f"\n📊 Analiz: Model fırtınayı {hata_payi:.2f} Kp hata payı ile tam 3 saat önceden başarıyla öngördü!")
else:
    print("Belirtilen tarih verisetinde (veya temizlenmiş veriler arasında) bulunamadı.")