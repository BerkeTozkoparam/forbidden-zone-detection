"""
╔══════════════════════════════════════════════════════════╗
║       YASAK BÖLGE İHLAL TESPİT SİSTEMİ                 ║
║       Forbidden Zone Intrusion Detection                 ║
║       YOLOv8 + OpenCV + Gerçekçi Simülasyon             ║
╚══════════════════════════════════════════════════════════╝

KULLANIM:
  python main.py --source sim                   # Simülasyon (varsayılan)
  python main.py --source sim --zone fixed      # Sabit bölge
  python main.py --source sim --entities 25     # Daha kalabalık
  python main.py --source 0                     # Webcam
  python main.py --source video.mp4             # Video

SİMÜLASYON KLAVYE:
  [S]  Yeni nesne ekle      [R]  Bölgeyi sıfırla
  [Q]  Çıkış
"""

import cv2
import numpy as np
import argparse
import time
import os
import random
import threading
from collections import deque
from datetime import datetime
from ultralytics import YOLO

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import joblib
    SENTINEL_OK = True
except ImportError:
    SENTINEL_OK = False

# ─────────────────────────────────────────────
#  YAPILANDIRMA
# ─────────────────────────────────────────────
CONFIG = {
    "model":          "yolov8n.pt",
    "confidence":     0.4,
    "target_classes": [0, 2, 3, 5, 7],
    "alert_color":    (0, 0, 255),
    "safe_color":     (0, 255, 0),
    "zone_color":     (0, 200, 255),
    "log_dir":        os.path.join(os.path.expanduser("~"), "Desktop", "ihlal_kayitlari"),
    "save_frames":    True,
    "sim_entities":   18,
    "sim_fps":        30,
}

CLASS_NAMES_TR = {
    0: "KISI", 2: "OTOMOBIL", 3: "MOTORSIKLET", 5: "OTOBUS", 7: "KAMYON"
}

# ─────────────────────────────────────────────
#  SENTINEL — IMU SENSOR FÜZYON
# ─────────────────────────────────────────────
IMU_CSV      = "data-2.csv"
IMU_FEATURES = ["accelX","accelY","accelZ",
                "linearAcX","linearAcY","linearAcZ",
                "orientX","orientY","orientZ"]
IMU_WINDOW   = 50    # pencere boyutu (örnek sayısı)
IMU_STEP     = 10    # eğitim sırasında adım
IMU_FPS      = 50    # CSV oynatma hızı (Hz)
_MODEL_FILE  = "sentinel_model.pkl"

# Tehdit renkleri (BGR) ve etiketleri
_THREAT_LEVELS = [
    (0.33, (0, 200,  80), "DUSUK"),
    (0.60, (0, 200, 255), "ORTA"),
    (0.80, (0, 100, 255), "YUKSEK"),
    (1.01, (0,   0, 255), "KRITIK"),
]

def _threat_style(level: float):
    for thr, color, label in _THREAT_LEVELS:
        if level < thr:
            return color, label
    return (0, 0, 255), "KRITIK"


def _extract_imu_features(window: np.ndarray) -> np.ndarray:
    """50×9 pencereden istatistiksel özellik vektörü çıkar."""
    feats = []
    for col in range(window.shape[1]):
        d = window[:, col]
        feats += [d.mean(), d.std(), d.min(), d.max()]
    # İvme ve lineer ivme büyüklükleri
    mag  = np.linalg.norm(window[:, :3], axis=1)
    lmag = np.linalg.norm(window[:, 3:6], axis=1)
    feats += [mag.mean(),  mag.std(),  mag.max()]
    feats += [lmag.mean(), lmag.std(), lmag.max()]
    return np.array(feats, dtype=np.float32)


class SentinelModel:
    """Random Forest (kimlik) + Isolation Forest (anomali) füzyon modeli."""

    def __init__(self, csv_path: str):
        if os.path.exists(_MODEL_FILE):
            print("  [*] Sentinel modeli yukleniyor...")
            self.clf, self.iso, self.scaler = joblib.load(_MODEL_FILE)  # type: ignore[name-defined]
            print("  [OK] Model yuklendi.")
        else:
            self.clf, self.iso, self.scaler = self._train(csv_path)
            joblib.dump((self.clf, self.iso, self.scaler), _MODEL_FILE)  # type: ignore[name-defined]
            print(f"  [OK] Model kaydedildi → {_MODEL_FILE}")

    def _train(self, csv_path: str):
        print("  [*] Dataset yukleniyor...")
        df = pd.read_csv(csv_path, index_col=0)  # type: ignore[name-defined]
        X_list, y_list = [], []
        for person, grp in df.groupby("class"):
            vals = grp[IMU_FEATURES].values
            for i in range(0, len(vals) - IMU_WINDOW, IMU_STEP):
                X_list.append(_extract_imu_features(vals[i:i + IMU_WINDOW]))
                y_list.append(person)
        X = np.array(X_list)
        y = np.array(y_list)
        print(f"  [OK] {len(X)} pencere olusturuldu.")

        scaler = StandardScaler().fit(X)  # type: ignore[name-defined]
        Xs = scaler.transform(X)

        print("  [*] Siniflandirici egitiliyor...")
        Xtr, Xte, ytr, yte = train_test_split(  # type: ignore[name-defined]
            Xs, y, test_size=0.2, stratify=y, random_state=42)
        clf = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)  # type: ignore[name-defined]
        clf.fit(Xtr, ytr)
        print(f"  [OK] Dogruluk: %{clf.score(Xte, yte)*100:.1f}")

        print("  [*] Anomali dedektoru egitiliyor...")
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)  # type: ignore[name-defined]
        iso.fit(Xs)
        return clf, iso, scaler

    def predict(self, window: np.ndarray):
        """Dönüş: (kisi_adi, guven_0_1, anomali_0_1)"""
        feat = _extract_imu_features(window).reshape(1, -1)
        feat_sc = self.scaler.transform(feat)

        probs  = self.clf.predict_proba(feat_sc)[0]
        person = self.clf.classes_[probs.argmax()]
        conf   = float(probs.max())

        iso_sc  = float(self.iso.score_samples(feat_sc)[0])
        # score_samples: daha negatif → daha anomali; [-0.3, 0.1] arası tipik
        anomaly = float(np.clip((-iso_sc - 0.05) / 0.25, 0.0, 1.0))
        return person, conf, anomaly


class IMUPlayer:
    """CSV'yi arka planda gerçek zamanlı sensor gibi oynatır."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, index_col=0)  # type: ignore[name-defined]
        self._data    = df[IMU_FEATURES].values
        self._idx     = 0
        self._buf     = deque(maxlen=IMU_WINDOW)  # type: ignore[name-defined]
        self._lock    = threading.Lock()
        self._window  = None
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        interval = 1.0 / IMU_FPS
        while self._running:
            t0 = time.time()
            with self._lock:
                self._buf.append(self._data[self._idx])
                self._idx = (self._idx + 1) % len(self._data)
                if len(self._buf) == IMU_WINDOW:
                    self._window = np.array(self._buf)
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    def get(self):
        with self._lock:
            return self._window

    def stop(self):
        self._running = False

# ─────────────────────────────────────────────
#  SAHNE SABİTLERİ  (1280 × 720, sabit)
# ─────────────────────────────────────────────
SW, SH = 1280, 720

# Y bantları
BLD_T0, BLD_T1 =   0,  72   # Üst bina cephesi
SWK_T0, SWK_T1 =  72, 114   # Üst kaldırım
RD_T0,  RD_T1  = 114, 210   # Üst yol  (2 şerit)
MID_T0, MID_T1 = 210, 250   # Geçiş bandı
OPN_Y0, OPN_Y1 = 250, 470   # Açık alan / plaza
MID_B0, MID_B1 = 470, 510   # Geçiş bandı
RD_B0,  RD_B1  = 510, 606   # Alt yol  (2 şerit)
SWK_B0, SWK_B1 = 606, 648   # Alt kaldırım
BLD_B0, BLD_B1 = 648,  SH   # Alt bina cephesi

# Şerit Y merkezleri
LY_TR = 141     # Üst yol → sağa
LY_TL = 183     # Üst yol ← sola
LY_BR = 537     # Alt yol → sağa
LY_BL = 579     # Alt yol ← sola
LY_ST =  92     # Üst kaldırım ortası
LY_SB = 627     # Alt kaldırım ortası

# Açık alan gezinti sınırları (biraz kenar boşluğu ile)
OX1, OX2 = 20, SW - 20
OY1, OY2 = OPN_Y0 + 12, OPN_Y1 - 12

# ─────────────────────────────────────────────
#  ARAÇ / YAYA SABİTLERİ
# ─────────────────────────────────────────────
# Araç boyutları {cls_id: (genişlik=sürüş yönü, yükseklik=yanal)}
VEH_DIMS = {2: (58, 27), 3: (38, 17), 5: (92, 34), 7: (88, 36)}

VEH_PALETTES = {
    2: [(28,75,175),(155,30,30),(28,135,30),(75,75,80),
        (195,175,45),(168,168,172),(55,125,158),(160,90,30)],
    3: [(45,45,48),(175,75,20),(28,28,145),(128,38,128),(200,180,20)],
    5: [(20,58,135),(135,95,28),(48,125,48),(60,60,100),(180,60,40)],
    7: [(48,75,115),(88,58,38),(58,58,62),(100,58,28),(38,80,78)],
}

PED_PALETTE = [
    (175,55,55),(55,55,175),(55,145,55),(145,95,55),
    (115,55,145),(55,125,125),(195,125,55),(95,95,95),
    (175,175,55),(55,95,175),(125,55,55),(75,125,75),
    (200,80,120),(80,160,200),(140,130,60),(60,80,140),
]


# ─────────────────────────────────────────────
#  YARDIMCI ÇIZIM FONKSİYONLARI
# ─────────────────────────────────────────────
def _add_noise(canvas, y0, y1, mag=7, rng=None):
    h = y1 - y0
    if h <= 0:
        return
    if rng is not None:
        n = rng.integers(-mag, mag + 1, (h, canvas.shape[1], 3), dtype=np.int16)
    else:
        n = np.random.randint(-mag, mag + 1, (h, canvas.shape[1], 3), dtype=np.int16)
    canvas[y0:y1] = np.clip(canvas[y0:y1].astype(np.int16) + n, 0, 255).astype(np.uint8)


def _draw_tree(canvas, tx, ty, rng):
    H, W = canvas.shape[:2]
    r = int(rng.integers(14, 22))
    dark  = (int(rng.integers(20,36)), int(rng.integers(82,110)), int(rng.integers(20,36)))
    light = (min(255, dark[0]+10), min(255, dark[1]+22), min(255, dark[2]+10))
    # Gövde
    cv2.rectangle(canvas,
                  (max(0,tx-3), max(0,ty-4)), (min(W-1,tx+3), min(H-1,ty+8)),
                  (40, 26, 12), -1)
    # Taç daireleri
    for ox, oy, rf, col in [
        (0,         -r+3,       r,          dark),
        (-int(r*.5),-int(r*.55),int(r*.7),  light),
        (int(r*.5), -int(r*.55),int(r*.7),  light),
        (0,         -int(r*1.3),int(r*.62), light),
    ]:
        cx2, cy2 = tx+ox, ty+oy
        if 0 <= cx2 < W and 0 <= cy2 < H and rf > 0:
            cv2.circle(canvas, (cx2, cy2), rf, col, -1)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ─────────────────────────────────────────────
#  SİMÜLASYON — NESNE
# ─────────────────────────────────────────────
class SimEntity:
    """
    mode:
      'veh_r'   → araç şeridinde sağa gidiyor
      'veh_l'   ← araç şeridinde sola gidiyor
      'ped_sw'  kaldırım yayası
      'ped_op'  açık alan yayası  (ihlal potansiyeli var)
    """

    def __init__(self, eid, mode=None):
        self.id = eid
        self.alive = True
        self.frame_count = 0
        self.anim_phase  = random.uniform(0, 6.28)

        if mode is None:
            mode = random.choices(
                ['veh_r', 'veh_l', 'ped_sw', 'ped_op'],
                weights=[16, 16, 22, 46]
            )[0]
        self.mode = mode
        self._init()

    # ── başlatma ─────────────────────────────
    def _init(self):
        m = self.mode
        if 'veh' in m:
            self.cls_id = random.choices([2, 3, 5, 7], weights=[55, 20, 15, 10])[0]
            bw, bh = VEH_DIMS[self.cls_id]
            self.w = bw + random.randint(-4, 4)
            self.h = bh + random.randint(-2, 2)
            self.color = random.choice(VEH_PALETTES[self.cls_id])

            if m == 'veh_r':
                lane_y = random.choice([LY_TR, LY_BR])
                self.x  = float(-self.w - 8)
                self.vx = random.uniform(2.5, 5.5)
            else:
                lane_y = random.choice([LY_TL, LY_BL])
                self.x  = float(SW + 8)
                self.vx = -random.uniform(2.5, 5.5)

            self.y  = float(lane_y - self.h // 2) + random.uniform(-2, 2)
            self.vy = 0.0
            self.dir_timer = 9999    # araç yön değiştirmez

        else:  # yaya
            self.cls_id = 0
            sz = random.randint(10, 15)
            self.w = sz
            self.h = int(sz * 1.65)
            self.color = random.choice(PED_PALETTE)

            if m == 'ped_sw':
                side = random.choice(['t', 'b'])
                base_y = LY_ST if side == 't' else LY_SB
                self.y  = float(base_y - self.h // 2) + random.uniform(-6, 6)
                go_r   = random.random() > 0.5
                self.x  = float(-self.w) if go_r else float(SW)
                self.vx = random.uniform(0.7, 1.6) * (1 if go_r else -1)
                self.vy = 0.0
            else:  # ped_op
                self.x = float(random.uniform(OX1, OX2 - self.w))
                self.y = float(random.uniform(OY1, OY2 - self.h))
                spd = random.uniform(0.5, 1.5)
                ang = random.uniform(0, 6.28)
                self.vx = spd * np.cos(ang)
                self.vy = spd * np.sin(ang)

            self.dir_timer = random.randint(50, 170)

    # ── hareket ──────────────────────────────
    def update(self):
        self.frame_count += 1
        self.x += self.vx
        self.y += self.vy

        m = self.mode
        if 'veh' in m:
            if self.vx > 0 and self.x > SW + self.w + 20:  self.alive = False
            if self.vx < 0 and self.x < -self.w - 20:      self.alive = False
        elif m == 'ped_sw':
            if self.x > SW + 20 or self.x < -self.w - 20:  self.alive = False
        else:  # ped_op
            # Sektir
            if   self.x < OX1:          self.vx =  abs(self.vx); self.x = OX1
            elif self.x + self.w > OX2: self.vx = -abs(self.vx); self.x = OX2 - self.w
            if   self.y < OY1:          self.vy =  abs(self.vy); self.y = OY1
            elif self.y + self.h > OY2: self.vy = -abs(self.vy); self.y = OY2 - self.h

            # Yön sapması
            self.dir_timer -= 1
            if self.dir_timer <= 0:
                self.dir_timer = random.randint(50, 170)
                spd = np.hypot(self.vx, self.vy) or 1.0
                ang = np.arctan2(self.vy, self.vx) + random.uniform(-1.2, 1.2)
                self.vx = spd * np.cos(ang)
                self.vy = spd * np.sin(ang)

    # ── sınır yardımcıları ────────────────────
    def get_bbox(self):
        return int(self.x), int(self.y), int(self.x + self.w), int(self.y + self.h)

    def get_foot_point(self):
        return int(self.x + self.w / 2), int(self.y + self.h)

    # ── çizim ────────────────────────────────
    def draw(self, frame):
        if 'veh' in self.mode:
            self._draw_vehicle(frame)
        else:
            self._draw_ped(frame)

    def _draw_vehicle(self, frame):
        x1, y1, x2, y2 = self.get_bbox()
        H, W = frame.shape[:2]
        going_r = self.vx > 0
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 3:
            return
        x1c = _clamp(x1, 0, W-1); y1c = _clamp(y1, 0, H-1)
        x2c = _clamp(x2, 0, W-1); y2c = _clamp(y2, 0, H-1)
        if x2c <= x1c or y2c <= y1c:
            return

        c = self.color

        # Gövde
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), c, -1)

        # Çatı (biraz daha açık)
        rc = tuple(min(255, v + 38) for v in c)
        mx, my = max(4, bw // 7), max(2, bh // 5)
        rx1 = _clamp(x1+mx, 0, W-1); ry1 = _clamp(y1+my, 0, H-1)
        rx2 = _clamp(x2-mx, 0, W-1); ry2 = _clamp(y2-my, 0, H-1)
        if rx2 > rx1 and ry2 > ry1:
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), rc, -1)

        # Ön cam (windshield)
        if going_r:
            wx1, wx2 = x1 + int(bw * .56), x2 - max(4, bw // 10)
        else:
            wx1, wx2 = x1 + max(4, bw // 10), x1 + int(bw * .44)
        wy1, wy2 = y1 + my + 1, y2 - my - 1
        wc1 = (_clamp(wx1,0,W-1), _clamp(wy1,0,H-1))
        wc2 = (_clamp(wx2,0,W-1), _clamp(wy2,0,H-1))
        if wc2[0] > wc1[0] and wc2[1] > wc1[1]:
            cv2.rectangle(frame, wc1, wc2, (152, 192, 210), -1)
            # Cam yansıması
            cv2.line(frame, wc1, (wc1[0], wc2[1]), (175, 210, 225), 1)

        # Far (ön) ve stop lambası (arka)
        lr = max(2, bh // 5)
        front_x = _clamp(x2-3 if going_r else x1+3, 0, W-1)
        rear_x  = _clamp(x1+3 if going_r else x2-3, 0, W-1)
        for ly in [y1 + bh // 4, y2 - bh // 4]:
            if 0 <= ly < H:
                cv2.circle(frame, (front_x, ly), lr,       (255, 252, 195), -1)
                cv2.circle(frame, (front_x, ly), max(1,lr-1),(255, 255, 220), 1)
                cv2.circle(frame, (rear_x,  ly), max(1,lr-1),(185, 28, 28),  -1)

        # Plaka
        py = (y1 + y2) // 2
        if 0 <= py < H:
            pxc = _clamp(front_x, 6, W-6)
            cv2.rectangle(frame, (pxc-7, py-3), (pxc+7, py+3), (228, 228, 145), -1)
            cv2.rectangle(frame, (pxc-7, py-3), (pxc+7, py+3), (90, 90, 70), 1)

        # Tekerlekler (4 köşe)
        ww, wh = max(3, bh // 5), max(2, bh // 9)
        for wx, wy in [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]:
            wxc = _clamp(wx, 0, W-1); wyc = _clamp(wy, 0, H-1)
            cv2.ellipse(frame, (wxc, wyc), (ww, wh), 0, 0, 360, (10,10,10), -1)
            cv2.ellipse(frame, (wxc, wyc), (max(1,ww-2), max(1,wh-1)), 0, 0, 360, (42,42,42), 1)

        # Dış hat
        cv2.rectangle(frame, (x1c, y1c), (x2c, y2c), (50, 50, 50), 1)

    def _draw_ped(self, frame):
        x1, y1, x2, y2 = self.get_bbox()
        pcx, pcy = (x1+x2)//2, (y1+y2)//2
        H, W = frame.shape[:2]
        if not (0 <= pcx < W and 0 <= pcy < H):
            return

        c = self.color
        phase = self.anim_phase + self.frame_count * 0.13
        sway  = int(np.sin(phase) * 2)

        # Gölge
        cv2.ellipse(frame, (pcx+sway, _clamp(pcy+5, 0, H-1)),
                    (self.w//2+2, 3), 0, 0, 360, (6,6,6), -1)
        # Gövde
        br = max(4, self.w // 2)
        cv2.ellipse(frame, (pcx, pcy), (br, int(br*1.35)), 0, 0, 360, c, -1)
        # Kafa
        hr  = max(3, self.w // 3)
        hcy = pcy - int(br * 1.32)
        if 0 <= hcy < H:
            skin = (min(255,182+sway*4), min(255,142+sway*2), 105)
            cv2.circle(frame, (pcx+sway//2, hcy), hr, skin, -1)
            cv2.circle(frame, (pcx+sway//2, hcy), hr, (48,30,20), 1)
        # Kıyafet aksesuarı (rastgele küçük ışık noktası → cep telefonu vs.)
        if self.id % 4 == 0:
            cv2.circle(frame, (pcx, pcy), 2, (200,220,255), -1)


# ─────────────────────────────────────────────
#  SİMÜLASYON — MOTOR
# ─────────────────────────────────────────────
class SimulationEngine:

    def __init__(self, n_entities=18):
        self.entities: list[SimEntity] = []
        self.eid = 0
        self.n   = n_entities
        self.spawn_timer = 0
        self._build_bg()
        for _ in range(n_entities):
            self._spawn()

    # ── arka plan inşası ─────────────────────
    def _build_bg(self):
        rng = np.random.default_rng(42)   # sabit tohum → tutarlı bina düzeni
        bg  = np.full((SH, SW, 3), 22, dtype=np.uint8)

        def fill(y0, y1, color):
            cv2.rectangle(bg, (0, y0), (SW, y1), color, -1)

        # ── Bina cepheleri (üst + alt) ────────
        for y0, y1 in [(BLD_T0, BLD_T1), (BLD_B0, BLD_B1)]:
            fill(y0, y1, (24, 24, 30))
            x = 0
            while x < SW:
                bw = int(rng.integers(55, 125))
                bh = int(rng.integers(20, max(22, y1-y0-5)))
                bc = (int(rng.integers(30,50)),int(rng.integers(30,50)),int(rng.integers(36,58)))
                bx2 = min(SW, x+bw)
                cv2.rectangle(bg, (x, y0), (bx2, y1), bc, -1)
                cv2.line(bg, (x, y0), (x, y1), (16,16,22), 1)
                # Pencereler
                for wx in range(x+8, bx2-6, 20):
                    for wy in range(y0+6, y0+bh-2, 18):
                        if wy+10 < y1:
                            p = rng.random()
                            if p > 0.28:
                                wc = (int(rng.integers(188,220)), int(rng.integers(172,208)), int(rng.integers(75,118))) if p > 0.72 \
                                     else (int(rng.integers(28,48)), int(rng.integers(28,48)), int(rng.integers(36,58)))
                                cv2.rectangle(bg, (wx, wy), (wx+12, wy+9), wc, -1)
                x += bw + int(rng.integers(0,4))

        # ── Kaldırımlar ───────────────────────
        for y0, y1 in [(SWK_T0, SWK_T1), (SWK_B0, SWK_B1)]:
            fill(y0, y1, (120, 115, 106))
            for sx in range(0, SW, 52):
                cv2.line(bg, (sx, y0), (sx, y1), (108, 103, 95), 1)
            for sy in range(y0, y1, 52):
                cv2.line(bg, (0, sy), (SW, sy), (108, 103, 95), 1)
            _add_noise(bg, y0, y1, 6, rng)

        # ── Yollar ────────────────────────────
        for y0, y1 in [(RD_T0, RD_T1), (RD_B0, RD_B1)]:
            fill(y0, y1, (38, 38, 38))
            _add_noise(bg, y0, y1, 5, rng)
            mid_y = (y0 + y1) // 2
            # Yol kenar çizgileri
            cv2.line(bg, (0, y0), (SW, y0), (222, 222, 222), 2)
            cv2.line(bg, (0, y1), (SW, y1), (222, 222, 222), 2)
            # Orta şerit (beyaz kesik)
            for x in range(0, SW, 78):
                cv2.line(bg, (x, mid_y), (min(SW, x+44), mid_y), (212, 212, 212), 2)

        # ── Geçiş bantları ────────────────────
        for y0, y1 in [(MID_T0, MID_T1), (MID_B0, MID_B1)]:
            fill(y0, y1, (110, 106, 97))
            _add_noise(bg, y0, y1, 5, rng)

        # ── Açık alan / Plaza ─────────────────
        fill(OPN_Y0, OPN_Y1, (80, 76, 70))
        # Beton levha ızgarası
        for gx in range(0, SW, 128):
            cv2.line(bg, (gx, OPN_Y0), (gx, OPN_Y1), (72, 68, 63), 1)
        for gy in range(OPN_Y0, OPN_Y1, 128):
            cv2.line(bg, (0, gy), (SW, gy), (72, 68, 63), 1)
        _add_noise(bg, OPN_Y0, OPN_Y1, 4, rng)

        # Zemin leke / yağ izi
        for _ in range(22):
            sx = int(rng.integers(40, SW-40))
            sy = int(rng.integers(OPN_Y0+8, OPN_Y1-8))
            sr = int(rng.integers(5, 22))
            ov = bg.copy()
            cv2.ellipse(ov, (sx,sy), (sr, max(2,sr//3)), int(rng.integers(0,180)),
                        0, 360, (65,62,57), -1)
            cv2.addWeighted(ov, 0.45, bg, 0.55, 0, bg)

        # ── Ağaçlar ───────────────────────────
        mid_t_y = (MID_T0 + MID_T1) // 2
        mid_b_y = (MID_B0 + MID_B1) // 2
        for tx in range(40, SW, 112):
            _draw_tree(bg, tx,     mid_t_y, rng)
            _draw_tree(bg, tx+56, mid_b_y, rng)

        # ── Zebra geçitleri ───────────────────
        for cw_x in [148, SW-148]:
            for y0, y1 in [(RD_T0, RD_T1), (RD_B0, RD_B1)]:
                for sy in range(y0, y1, 16):
                    if (sy // 16) % 2 == 0:
                        cv2.rectangle(bg,
                                      (_clamp(cw_x-16,0,SW-1), sy),
                                      (_clamp(cw_x+16,0,SW-1), min(y1, sy+11)),
                                      (212, 212, 212), -1)

        # ── Menhol kapakları ──────────────────
        for lane_y in [LY_TR, LY_TL, LY_BR, LY_BL]:
            for mx in range(260, SW-260, 330):
                cv2.circle(bg, (mx, lane_y), 9, (31,31,31), -1)
                cv2.circle(bg, (mx, lane_y), 9, (46,46,46), 1)
                cv2.line(bg, (mx-6,lane_y), (mx+6,lane_y), (46,46,46), 1)
                cv2.line(bg, (mx,lane_y-6), (mx,lane_y+6), (46,46,46), 1)

        # ── Yol levha direkleri ───────────────
        for pole_x in [200, 600, 1000]:
            px = pole_x
            for py in [RD_T0-2, RD_B1+2]:
                cv2.line(bg, (px, py), (px, py - (25 if py < SH//2 else -25)), (70,70,75), 2)
                cv2.circle(bg, (px, py - (25 if py < SH//2 else -25)), 5, (255,230,120), -1)

        # ── Güvenlik kulübesi ─────────────────
        kx1, ky1 = SW-95, OPN_Y0+5
        kx2, ky2 = SW-32, OPN_Y0+58
        cv2.rectangle(bg, (kx1, ky1), (kx2, ky2), (48, 68, 88), -1)
        cv2.rectangle(bg, (kx1, ky1), (kx2, ky2), (32, 48, 62), 1)
        # Cam
        cv2.rectangle(bg, (kx1+5, ky1+4), (kx2-5, ky1+24), (115, 178, 205), -1)
        cv2.line(bg, (kx1+5, ky1+4), (kx2-5, ky1+24), (80, 140, 165), 1)
        # Çatı
        roof = np.array([(kx1-4,ky1),(kx2+4,ky1),(kx2+4,ky1-10),(kx1-4,ky1-10)],np.int32)
        cv2.fillPoly(bg, [roof], (38, 52, 68))

        # ── Bariyer / Beton bloklar ───────────
        for bx in [160, 480, 800, 1120]:
            by_top = OPN_Y0 + 8
            cv2.rectangle(bg, (bx-14, by_top), (bx+14, by_top+24), (76,73,68), -1)
            cv2.rectangle(bg, (bx-14, by_top), (bx+14, by_top+24), (58,56,52), 1)
            # Gölge üst
            cv2.line(bg, (bx-14, by_top), (bx+14, by_top), (90,87,82), 2)

        # ── Uyarı çizgisi (açık alan girişi) ─
        cv2.line(bg, (0, OPN_Y0), (SW, OPN_Y0), (0, 180, 255), 1)
        cv2.line(bg, (0, OPN_Y1), (SW, OPN_Y1), (0, 180, 255), 1)

        # ── Otopark şeritleri (açık alanın alt yarısı) ──
        park_y0 = (OPN_Y0 + OPN_Y1) // 2 + 20
        for px in range(80, SW-80, 80):
            cv2.line(bg, (px, park_y0), (px, OPN_Y1-10), (65,62,58), 1)
        for prow in [park_y0, (park_y0 + OPN_Y1-10)//2, OPN_Y1-10]:
            cv2.line(bg, (80, prow), (SW-80, prow), (65,62,58), 1)

        self.background = bg

    # ── entity yönetimi ──────────────────────
    def _spawn(self, mode=None):
        e = SimEntity(self.eid, mode)
        self.entities.append(e)
        self.eid += 1

    def update(self):
        for e in self.entities:
            e.update()
        self.entities = [e for e in self.entities if e.alive]
        self.spawn_timer += 1
        if self.spawn_timer >= 20 and len(self.entities) < self.n:
            self._spawn()
            self.spawn_timer = 0

    # ── kare üretimi ─────────────────────────
    def render_frame(self):
        frame = self.background.copy()

        # Gölgeler (nesnelerin altına)
        for e in self.entities:
            x1, y1, x2, y2 = e.get_bbox()
            cx, cy = (x1+x2)//2, (y1+y2)//2
            H, W = frame.shape[:2]
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            if 'veh' in e.mode:
                sw = _clamp((x2-x1)//2+4, 4, W//2)
                cv2.ellipse(frame, (cx, _clamp(cy+4,0,H-1)),
                            (sw, 5), 0, 0, 360, (6,6,6), -1)
            else:
                cv2.ellipse(frame, (cx, _clamp(cy+5,0,H-1)),
                            (e.w//2+2, 3), 0, 0, 360, (6,6,6), -1)

        for e in self.entities:
            e.draw(frame)

        # Nesne sayısı
        cv2.putText(frame, f"Nesne: {len(self.entities)}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 120, 120), 1)
        return frame

    def get_detections(self):
        dets = []
        for e in self.entities:
            x1, y1, x2, y2 = e.get_bbox()
            if x2 < 0 or x1 > SW or y2 < 0 or y1 > SH:
                continue
            cx, cy = e.get_foot_point()
            dets.append({
                "bbox": (x1, y1, x2, y2),
                "cls_id": e.cls_id,
                "conf": round(random.uniform(0.82, 0.99), 2),
                "track_id": e.id,
                "cx": cx, "cy": cy,
            })
        return dets


# ─────────────────────────────────────────────
#  BÖLGE YÖNETİCİSİ
# ─────────────────────────────────────────────
class ZoneManager:
    def __init__(self):
        self.points    = []
        self.zone_ready = False
        self.temp_mouse = (0, 0)

    def mouse_callback(self, event, x, y, flags, param):
        self.temp_mouse = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            print(f"  [+] Nokta: ({x},{y})  —  Toplam: {len(self.points)}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.finish()

    def finish(self):
        """Bölgeyi tamamla — sağ tık veya Enter ile çağrılır."""
        if len(self.points) >= 3:
            self.zone_ready = True
            print(f"  [OK] Bolge tamamlandi. {len(self.points)} nokta.")
        else:
            print(f"  [!] En az 3 nokta gerekli! (Simdi: {len(self.points)})")

    def set_fixed_zone(self, fw, fh, sim_mode=False):
        if sim_mode:
            # Açık alan bandına yerleştir
            mx = int(fw * 0.22)
            y1 = OPN_Y0 + 18
            y2 = OPN_Y1 - 18
        else:
            mx = int(fw * 0.25)
            my = int(fh * 0.20)
            y1, y2 = my, fh - my
        self.points = [(mx,y1),(fw-mx,y1),(fw-mx,y2),(mx,y2)]
        self.zone_ready = True
        print("  [OK] Sabit bolge ayarlandi.")

    def is_inside(self, cx, cy):
        if not self.zone_ready or len(self.points) < 3:
            return False
        poly = np.array(self.points, dtype=np.int32)
        return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

    def draw_zone(self, frame):
        if len(self.points) < 2:
            return frame
        pts = np.array(self.points, dtype=np.int32)

        # Yarı saydam dolgu
        ov = frame.copy()
        if self.zone_ready:
            cv2.fillPoly(ov, [pts], (0, 35, 175))
        cv2.addWeighted(ov, 0.17, frame, 0.83, 0, frame)

        # Nabız efekti (zone hazırsa)
        if self.zone_ready:
            t = time.time()
            pulse_w = 2 + int(abs(np.sin(t * 2.0)) * 2)
            color   = (80, 190, 255)
            cv2.polylines(frame, [pts], True, color, pulse_w, cv2.LINE_AA)
            # Merkez etiketi
            cx2 = int(np.mean([p[0] for p in self.points]))
            cy2 = int(np.mean([p[1] for p in self.points]))
            (tw, th), _ = cv2.getTextSize("YASAK BOLGE", cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)
            cv2.rectangle(frame, (cx2-tw//2-6, cy2-th-8), (cx2+tw//2+6, cy2+6), (0,0,0), -1)
            cv2.putText(frame, "YASAK BOLGE", (cx2-tw//2, cy2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 200, 255), 2)
        else:
            # Tüm noktalar arası çizgiler
            for i in range(len(self.points) - 1):
                cv2.line(frame, self.points[i], self.points[i+1],
                         CONFIG["zone_color"], 2, cv2.LINE_AA)
            # Son noktadan mouse'a çizgi (temp_mouse (0,0) değilse)
            if self.points and self.temp_mouse != (0, 0):
                cv2.line(frame, self.points[-1], self.temp_mouse,
                         (110, 110, 110), 1, cv2.LINE_AA)

        for pt in self.points:
            col = (80,190,255) if self.zone_ready else CONFIG["zone_color"]
            cv2.circle(frame, pt, 6, col, -1)
            cv2.circle(frame, pt, 8, (255,255,255), 1)

        return frame


# ─────────────────────────────────────────────
#  İHLAL KAYIT SİSTEMİ
# ─────────────────────────────────────────────
class ViolationLogger:
    def __init__(self):
        frames_dir = os.path.join(CONFIG["log_dir"], "frames")
        os.makedirs(frames_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(CONFIG["log_dir"], f"violations_{ts}.txt")
        self.csv_file = os.path.join(CONFIG["log_dir"], f"violations_{ts}.csv")
        self.total    = 0
        self._cd: dict = {}   # cooldown
        # CSV başlık satırı
        with open(self.csv_file, "w", encoding="utf-8") as f:
            f.write("ihlal_no,tarih,saat,nesne_tipi,track_id,guven\n")

    def log(self, frame, track_id, cls_name, conf):
        now = time.time()
        key = f"{track_id}_{cls_name}"
        if key in self._cd and now - self._cd[key] < 3.0:
            return False
        self._cd[key] = now
        self.total += 1
        dt   = datetime.now()
        ts   = dt.strftime("%Y-%m-%d %H:%M:%S")
        line = (f"[{ts}] IHLAL #{self.total} | {cls_name} "
                f"| ID:{track_id} | Guven:{conf:.2f}\n")
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line)
        # CSV satırı
        with open(self.csv_file, "a", encoding="utf-8") as f:
            f.write(f"{self.total},{dt.strftime('%Y-%m-%d')},{dt.strftime('%H:%M:%S')},"
                    f"{cls_name},{track_id},{conf:.2f}\n")
            
            fname = os.path.join(CONFIG["log_dir"], "frames",
                                 f"v{self.total:04d}_{dt.strftime('%H%M%S')}.jpg")
            cv2.imwrite(fname, frame)
        print(f"  >> {line.strip()}")
        return True


# ─────────────────────────────────────────────
#  VİDEO KLİP KAYDEDİCİ
# ─────────────────────────────────────────────
class VideoRecorder:
    """
    İhlal tespit edilince video klip kaydeder.
    Son ihlalden `tail_secs` saniye sonra kaydı otomatik bitirir.
    """

    CODEC = "mp4v"

    def __init__(self, log_dir: str, frame_w: int, frame_h: int,
                 fps: int = 30, tail_secs: float = 6.0):
        self.video_dir  = os.path.join(log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.fw         = frame_w
        self.fh         = frame_h
        self.fps        = fps
        self.tail       = tail_secs
        self._writer    = None
        self._active    = False
        self._last_vt   = 0.0
        self._clip_n    = 0
        self._cur_file  = ""

    @property
    def is_recording(self) -> bool:
        return self._active

    def notify_violation(self):
        """Her ihlal tespitinde çağrılır."""
        self._last_vt = time.time()
        if not self._active:
            self._start()

    def _start(self):
        self._clip_n += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._cur_file = os.path.join(
            self.video_dir,
            f"clip_{self._clip_n:04d}_{ts}.mp4"
        )
        fourcc = cv2.VideoWriter_fourcc(*self.CODEC)
        self._writer = cv2.VideoWriter(
            self._cur_file, fourcc, self.fps, (self.fw, self.fh)
        )
        self._active = True
        print(f"  [REC] Kayit basladi  → {os.path.basename(self._cur_file)}")

    def write(self, frame):
        """Her frame sonunda çağrılır."""
        if self._active and self._writer is not None:
            self._writer.write(frame)

    def tick(self):
        """Her frame'de çağrılır; kuyruk süresi bittiyse kaydı durdurur."""
        if self._active and time.time() - self._last_vt > self.tail:
            self._stop()

    def _stop(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._active = False
        print(f"  [REC] Kayit tamamlandi → {os.path.basename(self._cur_file)}")

    def close(self):
        if self._active:
            self._stop()


# ─────────────────────────────────────────────
#  ORTAK TESPİT İŞLEME
# ─────────────────────────────────────────────
def process_detections(frame, frame_copy, dets, zone, logger):
    violated_any = False
    for det in dets:
        x1, y1, x2, y2 = det["bbox"]
        cls_id   = det["cls_id"]
        conf     = det["conf"]
        track_id = det["track_id"]
        cx, cy   = det["cx"], det["cy"]

        violated = zone.is_inside(cx, cy)
        cls_name = CLASS_NAMES_TR.get(cls_id, f"Nesne-{cls_id}")

        if violated:
            violated_any = True
            color = CONFIG["alert_color"]
            label = f"!IHLAL! {cls_name} #{track_id}"
            logger.log(frame_copy, track_id, cls_name, conf)
        else:
            color = CONFIG["safe_color"]
            label = f"{cls_name} #{track_id} {conf:.2f}"

        _draw_box(frame, x1, y1, x2, y2, label, color, violated)
        dot_c = (0,0,255) if violated else (0,255,0)
        H, W = frame.shape[:2]
        if 0 <= cx < W and 0 <= cy < H:
            cv2.circle(frame, (cx, cy), 5, dot_c, -1)
            cv2.circle(frame, (cx, cy), 5, (255,255,255), 1)

    return violated_any


# ─────────────────────────────────────────────
#  UI YARDIMCILARI
# ─────────────────────────────────────────────
def _draw_box(frame, x1, y1, x2, y2, label, color, violated):
    H, W = frame.shape[:2]
    x1c = _clamp(x1,0,W-1); y1c = _clamp(y1,0,H-1)
    x2c = _clamp(x2,0,W-1); y2c = _clamp(y2,0,H-1)
    if x2c <= x1c or y2c <= y1c:
        return
    # İhlal kırmızı overlay
    if violated:
        ov = frame.copy()
        cv2.rectangle(ov, (x1c,y1c), (x2c,y2c), (0,0,200), -1)
        cv2.addWeighted(ov, 0.2, frame, 0.8, 0, frame)
    # Köşe bracket
    cl = 14
    for (cx2,cy2),(hpt),(vpt) in [
        ((x1c,y1c),(x1c+cl,y1c),(x1c,y1c+cl)),
        ((x2c,y1c),(x2c-cl,y1c),(x2c,y1c+cl)),
        ((x1c,y2c),(x1c+cl,y2c),(x1c,y2c-cl)),
        ((x2c,y2c),(x2c-cl,y2c),(x2c,y2c-cl)),
    ]:
        cv2.line(frame, (cx2,cy2), hpt, color, 2)
        cv2.line(frame, (cx2,cy2), vpt, color, 2)
    # Etiket arka planı
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
    bg_y1 = _clamp(y1c - th - 10, 0, H-1)
    cv2.rectangle(frame, (x1c, bg_y1), (_clamp(x1c+tw+8,0,W-1), y1c), color, -1)
    cv2.putText(frame, label, (x1c+4, _clamp(y1c-5, th+2, H-1)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)


def _draw_sentinel_panel(frame, sentinel: dict):
    """Sağ alt köşeye IMU füzyon paneli çizer."""
    H, W = frame.shape[:2]
    person  = sentinel.get("person", "?")
    conf    = sentinel.get("conf", 0.0)
    anomaly = sentinel.get("anomaly", 0.0)
    threat  = sentinel.get("threat", 0.0)
    tc, tlabel = _threat_style(threat)

    pw, ph = 210, 110
    x0, y0 = W - pw - 8, H - ph - 48   # alt şeridin üstünde

    # Arka plan
    ov = frame.copy()
    cv2.rectangle(ov, (x0, y0), (x0+pw, y0+ph), (10, 12, 10), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0+pw, y0+ph), tc, 1)

    # Başlık
    cv2.putText(frame, "IMU SENSOR FUZYON", (x0+6, y0+14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 80), 1)

    # Kişi profili
    cv2.putText(frame, f"PROFIL : {person.upper()}", (x0+6, y0+30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)
    cv2.putText(frame, f"GUVEN  : %{conf*100:.0f}", (x0+6, y0+45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)

    # Anomali bar
    cv2.putText(frame, "ANOMALI", (x0+6, y0+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (130, 130, 130), 1)
    bw = pw - 12
    cv2.rectangle(frame, (x0+6, y0+63), (x0+6+bw, y0+72), (35, 35, 35), -1)
    cv2.rectangle(frame, (x0+6, y0+63), (x0+6+int(bw*anomaly), y0+72), tc, -1)

    # Tehdit seviyesi
    cv2.putText(frame, f"TEHDIT : {tlabel}", (x0+6, y0+88),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, tc, 1)
    cv2.rectangle(frame, (x0+6, y0+91), (x0+6+bw, y0+100), (35, 35, 35), -1)
    cv2.rectangle(frame, (x0+6, y0+91), (x0+6+int(bw*threat), y0+100), tc, -1)


def draw_hud(frame, fps, total, zone_ready, draw_mode, alert_active, is_sim,
             is_recording=False, sentinel=None):
    H, W = frame.shape[:2]

    # Üst şerit
    cv2.rectangle(frame, (0,0), (W,45), (16,16,16), -1)
    cv2.rectangle(frame, (0,45), (W,47), (0,190,255), -1)
    tag = "[SIM]" if is_sim else "[CANLI]"
    tag_c = (0,200,100) if is_sim else (0,100,255)
    cv2.putText(frame, "YASAK BOLGE IHLAL TESPIT SISTEMI", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0,190,255), 2)
    cv2.putText(frame, tag, (W-90, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, tag_c, 2)
    cv2.putText(frame, f"{fps:.1f}fps", (W-165, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (145,145,145), 1)

    # REC göstergesi — yanıp sönen kırmızı nokta
    if is_recording and int(time.time() * 2) % 2 == 0:
        cv2.circle(frame, (W-220, 28), 7, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (W-208, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 255), 2)

    # Alt şerit
    cv2.rectangle(frame, (0,H-40), (W,H), (16,16,16), -1)
    vc = (0,80,255) if total > 0 else (0,200,100)
    cv2.putText(frame, f"IHLAL: {total}", (10, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, vc, 2)
    zt = "BOLGE: HAZIR" if zone_ready else "BOLGE: CIZILIYOR..."
    zc = (0,255,100) if zone_ready else (0,200,255)
    cv2.putText(frame, zt, (W//2-88, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, zc, 1)
    mt = "MOUSE" if draw_mode else "SABIT"
    cv2.putText(frame, f"MOD:{mt}", (W-155, H-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (130,130,130), 1)

    # Sentinel paneli
    if sentinel is not None:
        _draw_sentinel_panel(frame, sentinel)

    # Alarm çerçevesi
    if alert_active and int(time.time()*3) % 2 == 0:
        cv2.rectangle(frame, (2,48), (W-2,H-42), (0,0,255), 4)
        msg = "!! IHLAL ALGILANDI !!"
        (mw,_),_ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        cv2.putText(frame, msg, (W//2-mw//2, H//2-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
    return frame


def draw_instructions(frame, zone_ready, draw_mode, is_sim):
    if zone_ready:
        return frame
    H, W = frame.shape[:2]
    lines = ["[ TALIMATLAR ]","","Sol tik : Nokta ekle",
             "Sag tik / Enter : Bitir","  (min. 3 nokta)","","[R] Sifirla"]
    if is_sim:
        lines.append("[S] Nesne ekle")
    lines.append("[Q] Cikis")
    x0 = W - 192
    for i, txt in enumerate(lines):
        col = (0,190,255) if i==0 else (162,162,162)
        cv2.putText(frame, txt, (x0, 82+i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)
    return frame


# ─────────────────────────────────────────────
#  ANA DÖNGÜ
# ─────────────────────────────────────────────
def run(source: str, zone_mode: str):
    is_sim = source.lower() == "sim"

    print("\n" + "="*58)
    print("  YASAK BOLGE IHLAL TESPIT SISTEMI")
    print("="*58)
    print(f"  Kaynak : {'Simulasyon Modu (1280x720)' if is_sim else source}")
    print(f"  Bolge  : {'Mouse ile ciz' if zone_mode=='draw' else 'Sabit'}")
    if not is_sim:
        print(f"  Model  : {CONFIG['model']}")
    print("="*58 + "\n")

    # ── Kaynak hazırla ────────────────────────
    sim   = None
    cap   = None
    model = None
    frame_w, frame_h = SW, SH

    if is_sim:
        print("  [*] Sahne olusturuluyor...")
        sim = SimulationEngine(CONFIG["sim_entities"])
        print("  [OK] Simulasyon motoru hazir!\n")
    else:
        print("  [*] YOLOv8 yukleniyor...")
        model = YOLO(CONFIG["model"])
        print("  [OK] Model hazir!\n")
        src = int(source) if source.isdigit() else source
        _cap = cv2.VideoCapture(src)
        if not _cap.isOpened():
            print(f"  [HATA] Kaynak acilamadi: {source}")
            return
        cap = _cap
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  [OK] Cozunurluk: {frame_w}x{frame_h}")

    zone     = ZoneManager()
    logger   = ViolationLogger()
    recorder = VideoRecorder(CONFIG["log_dir"], frame_w, frame_h,
                             fps=CONFIG["sim_fps"])

    # ── Sentinel (IMU Füzyon) ──────────────────
    sentinel_model = None
    imu_player     = None
    if SENTINEL_OK and os.path.exists(IMU_CSV):
        print("  [*] Sentinel sistemi baslatiliyor...")
        sentinel_model = SentinelModel(IMU_CSV)
        imu_player     = IMUPlayer(IMU_CSV)
        time.sleep(0.5)   # buffer dolsun
        print("  [OK] IMU fuzyon aktif!\n")
    else:
        if not SENTINEL_OK:
            print("  [!] Sentinel devre disi (pandas/sklearn eksik)\n")
        elif not os.path.exists(IMU_CSV):
            print(f"  [!] Sentinel devre disi ({IMU_CSV} bulunamadi)\n")

    win = "Yasak Bolge Ihlal Tespit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(frame_w, 1280), min(frame_h, 720))

    # macOS: mouse callback pencere ilk kez gösterilmeden çalışmıyor
    _blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    cv2.imshow(win, _blank)
    cv2.waitKey(1)

    draw_mode = (zone_mode == "draw")
    if draw_mode:
        cv2.setMouseCallback(win, zone.mouse_callback)
        print("  [*] Mouse ile bolge cizin:")
        print("      Sol tik=nokta | Enter/Space=bitir\n")
    else:
        zone.set_fixed_zone(frame_w, frame_h, sim_mode=is_sim)

    fps_cnt, fps_start = 0, time.time()
    cur_fps = 0.0
    alert_active, alert_timer = False, 0.0
    wait_ms = max(1, int(1000 / CONFIG["sim_fps"])) if is_sim else 1

    print(f"  [*] {'SIMULASYON' if is_sim else 'CANLI'} mod basladi. [Q] cikis | [R] bolge sifirla")
    if is_sim:
        print("  [*] [S] = yeni nesne ekle\n")

    # ── Ana döngü ─────────────────────────────
    while True:
        if is_sim:
            assert sim is not None
            sim.update()
            frame = sim.render_frame()
        else:
            assert cap is not None
            ret, frame = cap.read()
            if not ret:
                if not source.isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

        frame_copy = frame.copy()
        frame = zone.draw_zone(frame)

        violated_frame = False
        if zone.zone_ready:
            if is_sim:
                assert sim is not None
                dets = sim.get_detections()
                violated_frame = process_detections(frame, frame_copy, dets, zone, logger)
            else:
                assert model is not None
                results = model.track(
                    frame, persist=True,
                    conf=CONFIG["confidence"],
                    classes=CONFIG["target_classes"],
                    verbose=False,
                )[0]
                if results.boxes is not None:
                    dets = []
                    for box in results.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        cls_id   = int(box.cls[0])
                        conf     = float(box.conf[0])
                        track_id = int(box.id[0]) if box.id is not None else -1
                        dets.append({"bbox":(x1,y1,x2,y2), "cls_id":cls_id,
                                     "conf":conf, "track_id":track_id,
                                     "cx":(x1+x2)//2, "cy":y2})
                    violated_frame = process_detections(frame, frame_copy, dets, zone, logger)

        if violated_frame:
            alert_active = True; alert_timer = time.time()
            recorder.notify_violation()
        elif time.time() - alert_timer > 2.0:
            alert_active = False

        recorder.tick()   # kuyruk süresi bittiyse kaydı durdurur

        # ── Sentinel füzyon ───────────────────
        sentinel_data = None
        if sentinel_model is not None and imu_player is not None:
            imu_win = imu_player.get()
            if imu_win is not None:
                person, conf, anomaly = sentinel_model.predict(imu_win)
                # Tehdit = anomali + düşük kimlik güveni + aktif ihlal
                threat = float(np.clip(
                    anomaly * 0.5 + (1.0 - conf) * 0.3 + (0.2 if violated_frame else 0.0),
                    0.0, 1.0
                ))
                sentinel_data = {
                    "person": person, "conf": conf,
                    "anomaly": anomaly, "threat": threat,
                }
                # Kritik tehdit → alarmı tetikle
                if threat > 0.80 and not alert_active:
                    alert_active = True; alert_timer = time.time()
                    recorder.notify_violation()

        fps_cnt += 1
        if fps_cnt >= 15:
            cur_fps = fps_cnt / (time.time() - fps_start)
            fps_cnt = 0; fps_start = time.time()

        frame = draw_hud(frame, cur_fps, logger.total, zone.zone_ready,
                         draw_mode, alert_active, is_sim,
                         is_recording=recorder.is_recording,
                         sentinel=sentinel_data)
        frame = draw_instructions(frame, zone.zone_ready, draw_mode, is_sim)

        recorder.write(frame)   # HUD dahil tam kareyi kaydet
        cv2.imshow(win, frame)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (13, ord(' ')) and not zone.zone_ready:
            # Enter veya Space → bölgeyi tamamla (macOS sağ tık alternatifi)
            zone.finish()
        elif key == ord('r'):
            zone.points = []; zone.zone_ready = False
            print("  [*] Bolge sifirlandi.")
        elif key == ord('s') and is_sim:
            assert sim is not None
            sim._spawn()
            print(f"  [+] Yeni nesne. Toplam: {len(sim.entities)}")

    recorder.close()
    if imu_player is not None:
        imu_player.stop()
    if not is_sim:
        assert cap is not None
        cap.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*58}")
    print("  OTURUM OZETI")
    print(f"{'='*58}")
    print(f"  Toplam Ihlal : {logger.total}")
    print(f"  Log Dosyasi  : {logger.log_file}")
    print(f"  CSV Raporu   : {logger.csv_file}")
    if CONFIG["save_frames"] and logger.total > 0:
        print(f"  Kaydedilen   : {os.path.join(CONFIG['log_dir'], 'frames')}/")
    if logger.total > 0:
        print(f"  Video Kayit  : {os.path.join(CONFIG['log_dir'], 'videos')}/")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────
#  GİRİŞ NOKTASI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yasak Bolge Ihlal Tespit Sistemi")
    parser.add_argument("--source",   type=str, default="sim",
                        help="Kaynak: 0 (webcam) | video.mp4 | sim")
    parser.add_argument("--zone",     type=str, default="draw",
                        choices=["draw","fixed"],
                        help="Bolge modu: draw | fixed")
    parser.add_argument("--conf",     type=float, default=0.4,
                        help="Guven esigi (0.0-1.0)")
    parser.add_argument("--entities", type=int,   default=18,
                        help="Simulasyon nesne sayisi")
    args = parser.parse_args()
    CONFIG["confidence"]   = args.conf
    CONFIG["sim_entities"] = args.entities

    run(args.source, args.zone)
