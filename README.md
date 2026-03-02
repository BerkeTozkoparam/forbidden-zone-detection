# Yasak Bölge İhlal Tespit Sistemi
### Forbidden Zone Intrusion Detection — YOLOv8 + OpenCV + IMU Sensor Fusion

Gerçek zamanlı yasak bölge ihlal tespit sistemi. Webcam, video dosyası veya dahili simülasyon modu ile çalışır. Görüntü işleme ile IMU sensör verilerini birleştiren **Sentinel** füzyon motoru sayesinde kimlik tespiti ve anomali analizi yapar.

---

## Özellikler

### Görüntü İşleme
- **YOLOv8** ile gerçek zamanlı nesne tespiti ve takibi
- **Mouse ile serbest poligon** çizimi (min. 3 nokta)
- **macOS uyumlu** — Enter/Space ile bölge tamamlama (sağ tık alternatifi)
- **Simülasyon modu** — webcam veya video olmadan test edilebilir
  - Gerçekçi üstten görünüm: yol şeritleri, kaldırım, bina cepheleri
  - Şerit takip eden araçlar (far, plaka, cam efekti)
  - Yürüme animasyonlu yayalar
  - Ağaçlar, zebra geçidi, güvenlik kulübesi, bariyer blokları

### Sentinel — IMU Sensör Füzyonu
- **Random Forest** ile kişi kimliği tespiti (%90.8 doğruluk, 6 kişi profili)
- **Isolation Forest** ile hareket anomali skoru
- **Füzyon formülü:** `anomali × 0.5 + (1 − kimlik_güveni) × 0.3 + ihlal × 0.2`
- Kritik tehdit (>0.80) → otomatik alarm + video kaydı tetikler
- Model ilk çalıştırmada eğitilir, `sentinel_model.pkl` olarak önbelleğe alınır

### Tehdit Isı Haritası
- Her tespitte konuma Gaussian blob eklenir
- İhlal bölgelerinde 4× yoğunluk
- `decay=0.97` ile zamanla solar
- `[H]` tuşuyla açma/kapama

### Sentinel Dashboard (sağ panel)
- Anlık tehdit seviyesi + renk göstergesi (DUSUK / ORTA / YUKSEK / KRITIK)
- Son 200 kare tehdit geçmişi grafiği
- **Taktik saha haritası** — ısı haritası, yasak bölge ve varlık noktaları
- IMU profil, güven ve anomali geçmişi
- Oturum istatistikleri: süre, ihlal sayısı, video klip sayısı

### Kayıt & Loglama
- **Otomatik video kaydı** — ihlal anında başlar, 6 sn sonra durur
- **Yanıp sönen REC** göstergesi
- `.txt` ihlal logu + `.csv` rapor (Excel/pandas uyumlu)
- İhlal anı görüntüsü (`frames/`)
- Tüm çıktılar `~/Desktop/ihlal_kayitlari/` klasörüne kaydedilir

---

## Kurulum

```bash
git clone https://github.com/BerkeTozkoparan/forbidden-zone-detection.git
cd forbidden-zone-detection
pip install -r requirements.txt
```

> YOLOv8 modeli (`yolov8n.pt`) ilk çalıştırmada otomatik indirilir.
> Sentinel modeli `data-2.csv` dosyası mevcutsa otomatik eğitilir.

---

## Kullanım

```bash
# Simülasyon modu (webcam/video gerekmez)
python main.py --source sim

# Simülasyon + sabit bölge (mouse çizmeden)
python main.py --source sim --zone fixed

# Simülasyon + daha kalabalık ortam
python main.py --source sim --entities 25

# Webcam
python main.py --source 0

# Video dosyası
python main.py --source video.mp4
```

### Argümanlar

| Argüman | Varsayılan | Açıklama |
|---|---|---|
| `--source` | `sim` | `sim` / `0` (webcam) / `video.mp4` |
| `--zone` | `draw` | `draw` (mouse) / `fixed` (sabit) |
| `--conf` | `0.4` | YOLOv8 güven eşiği (0.0–1.0) |
| `--entities` | `18` | Simülasyon nesne sayısı |

---

## Klavye Kısayolları

| Tuş | Eylem |
|---|---|
| `Sol tık` | Poligon noktası ekle |
| `Sağ tık` / `Enter` / `Space` | Bölgeyi tamamla (min. 3 nokta) |
| `H` | Isı haritasını aç/kapat |
| `S` | Simülasyona yeni nesne ekle |
| `R` | Bölgeyi sıfırla |
| `Q` / `Esc` | Çıkış |

---

## Sistem Mimarisi

```
┌─────────────────────────────────────────────────────────┐
│              SENSOR FÜZYON SİSTEMİ                      │
├──────────────────────────┬──────────────────────────────┤
│   IMU VERİSİ (CSV)       │   GÖRÜNTÜ (Sim / Webcam)     │
│                          │                              │
│  Random Forest           │  YOLOv8 Tespit + Takip       │
│  → Kimlik tespiti        │  → Bbox, sınıf, track ID     │
│                          │                              │
│  Isolation Forest        │  Poligon bölge kontrolü      │
│  → Anomali skoru         │  → İhlal / güvenli           │
│                          │                              │
├──────────────────────────┴──────────────────────────────┤
│              FÜZYON MOTORU                              │
│   tehdit = anomali×0.5 + (1−güven)×0.3 + ihlal×0.2    │
├─────────────────────────────────────────────────────────┤
│   ÇIKTI: Alarm · Video · CSV · Log · Isı Haritası       │
└─────────────────────────────────────────────────────────┘
```

---

## Simülasyon Sahnesi

```
┌────────────────────────────────────────┐
│  Bina cepheleri (pencere efekti)       │
├────────────────────────────────────────┤
│  Kaldırım — yayalar                    │
├────────────────────────────────────────┤
│  Yol  →  şerit 1                       │
│  Yol  ←  şerit 2   (zebra geçidi)     │
├────────────────────────────────────────┤
│  Ağaç sırası                           │
├────────────────────────────────────────┤
│                                        │
│   AÇIK ALAN / PLAZA                    │  ← Yasak bölge buraya konur
│   (güvenlik kulübesi, bariyerler)      │
│                                        │
├────────────────────────────────────────┤
│  Ağaç sırası                           │
├────────────────────────────────────────┤
│  Yol  →  şerit 3                       │
│  Yol  ←  şerit 4   (zebra geçidi)     │
├────────────────────────────────────────┤
│  Kaldırım — yayalar                    │
├────────────────────────────────────────┤
│  Bina cepheleri                        │
└────────────────────────────────────────┘
```

---

## Çıktı Dosyaları

```
~/Desktop/ihlal_kayitlari/
├── violations_YYYYMMDD_HHMMSS.txt   # Metin log
├── violations_YYYYMMDD_HHMMSS.csv   # CSV rapor (Excel/pandas)
├── frames/
│   ├── v0001_143022.jpg             # İhlal anı görüntüsü
│   └── ...
└── videos/
    ├── clip_0001_YYYYMMDD_HHMMSS.mp4  # İhlal video klibi
    └── ...
```

CSV formatı:
```
ihlal_no,tarih,saat,nesne_tipi,track_id,guven
1,2026-03-02,11:40:26,KISI,4,0.93
```

---

## Gereksinimler

```
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
pandas >= 1.5.0
scikit-learn >= 1.2.0
joblib >= 1.2.0
```

Python 3.10+ önerilir.

---

## Gelecek Planlar

- [ ] Gerçek IMU sensörü bağlantısı (Arduino / MPU-6050)
- [ ] LSTM tabanlı hareket anomali tespiti
- [ ] Kalman filtresi ile konum tahmini (ihlalden önce uyar)
- [ ] Radar görünümü modu
- [ ] Çoklu yasak bölge desteği (sarı/kırmızı perimeter)
- [ ] Web dashboard (Flask ile uzaktan izleme)
