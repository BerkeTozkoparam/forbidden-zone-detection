# 🚧 Yasak Bölge İhlal Tespit Sistemi
### Forbidden Zone Intrusion Detection — YOLOv8 + OpenCV

Gerçek zamanlı yasak bölge ihlal tespit sistemi. Webcam, video dosyası veya dahili simülasyon modu ile çalışır. Kullanıcı tarafından çizilen poligon bölgeye giren kişi ve araçları tespit ederek alarm üretir ve ihlalleri loglar.

---

## Özellikler

- **YOLOv8** ile gerçek zamanlı nesne tespiti ve takibi
- **Mouse ile serbest poligon** çizimi — kaç nokta olursa olsun
- **Simülasyon modu** — webcam veya video olmadan test edilebilir
  - Gerçekçi üstten görünüm: yol şeritleri, kaldırım, bina cepheleri
  - Şerit takip eden araçlar (far, plaka, cam efekti)
  - Yürüme animasyonlu yayalar
  - Ağaçlar, zebra geçidi, güvenlik kulübesi, bariyer bloklarl
- **İhlal loglama** — `.txt` log + anlık frame kayıt (`logs/frames/`)
- **Spam önleme** — aynı nesne 3 saniyede bir loglanır
- **HUD** — FPS, ihlal sayacı, bölge durumu, alarm efekti

---

## Kurulum

```bash
git clone https://github.com/BerkeTozkoparan/forbidden-zone-detection.git
cd forbidden-zone-detection
pip install -r requirements.txt
```

> YOLOv8 modeli (`yolov8n.pt`) ilk çalıştırmada otomatik indirilir.

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

# Güven eşiği ayarı
python main.py --source 0 --conf 0.5
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
| `Sağ tık` | Bölgeyi tamamla (min. 3 nokta) |
| `S` | Simülasyona yeni nesne ekle |
| `R` | Bölgeyi sıfırla |
| `Q` / `Esc` | Çıkış |

---

## Tespit Edilen Nesne Sınıfları

| Sınıf | Açıklama |
|---|---|
| Kişi | Yaya ihlalleri |
| Otomobil | Araç girişi |
| Motorsiklet | Motorsiklet girişi |
| Otobüs | Otobüs girişi |
| Kamyon | Kamyon girişi |

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
logs/
├── violations_YYYYMMDD_HHMMSS.txt   # Metin log
└── frames/
    ├── v0001_143022.jpg              # İhlal anı görüntüsü
    ├── v0002_143025.jpg
    └── ...
```

Log formatı:
```
[2026-03-01 22:56:48] IHLAL #1 | KISI | ID:3 | Guven:0.82
```

---

## Gelecek Planlar

- [ ] Radar görünümü modu (`--mode radar`)
- [ ] Gece modu simülasyonu
- [ ] Web dashboard (ihlal istatistikleri)
- [ ] E-posta / sesli alarm entegrasyonu
- [ ] Çoklu kamera desteği

---

## Gereksinimler

```
ultralytics >= 8.0.0
opencv-python >= 4.8.0
numpy >= 1.24.0
```

Python 3.10+ önerilir.
