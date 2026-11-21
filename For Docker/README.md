# Sistem Pemantauan Akustik Real-Time (Sawmill vs Non-Sawmill)

Aplikasi web pemantauan dan klasifikasi suara lingkungan secara kontinu dan real-time menggunakan model Machine Learning (**RandomForestClassifier**) untuk mendeteksi keberadaan aktivitas mesin gergaji kayu (*Sawmill*) versus kebisingan lingkungan normal (*Non-Sawmill*), dilengkapi dengan visualisasi osiloskop, analisis telemetri fitur audio Librosa, dan sistem peringatan otomatis ke WhatsApp via Webhook jika terdeteksi suara sawmill kontinu selama 10 detik atau lebih.

---

## 1. Persyaratan Sistem & Arsitektur

* **Platform Server Target:** Proxmox VE / Linux Server / Komputer Lokal
* **Container Engine:** Docker & Docker Compose
* **Port yang Dibuka:** **1 Port saja** (Port `5000`)
* **Spesifikasi Model:**
  * File Model: `model_terbaik_klasifikasi_audio.joblib`
  * Algoritma: `RandomForestClassifier` (Scikit-Learn)
  * Ukuran Model: **±58 KB** (Model sudah disertakan langsung di dalam repositori ini, tidak memerlukan unduhan eksternal).
* **Ekstraksi Fitur:** 74 Fitur Audio Librosa (MFCC 20 koefisien, Chroma STFT 12 nada, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate, RMS) dengan normalisasi adaptif (*Automatic Gain Control*).
* **Sliding Window:** Durasi analisis 2.5 detik per window dengan pergeseran *hop* 0.5 detik.

---

## 2. Cara Menjalankan Aplikasi (Deployment Docker)

Sesuai instruksi pengujian lab, aplikasi ini dapat langsung dibangun dan dijalankan hanya dengan **satu perintah**:

### Langkah 1: Kloning Repositori
```bash
git clone <URL_REPOSITORY_GITHUB_ANDA>
cd <NAMA_FOLDER_REPOSITORY>
```

### Langkah 2: Jalankan via Docker Compose
```bash
docker compose up --build
```
> **Catatan:** Docker akan otomatis mengunduh base image Python, memasang pustaka sistem Linux audio (`libsndfile1` & `ffmpeg`), memasang seluruh dependensi dari `requirements.txt`, dan menjalankan server Flask pada port 5000.

### Langkah 3: Akses Dashboard
Buka peramban web (*browser*) dan arahkan ke:
* **Lokal:** [http://localhost:5000](http://localhost:5000) atau [http://127.0.0.1:5000](http://127.0.0.1:5000)
* **Server Lab (Proxmox):** `http://<IP_SERVER_PROXMOX>:5000`

Untuk menghentikan kontainer:
```bash
docker compose down
```

---

## 3. Konfigurasi Lingkungan (Environment Variables)

Aplikasi telah dilengkapi proteksi kredensial menggunakan berkas `.env`. Template konfigurasi tersedia pada berkas `.env.example`:

| Variabel | Default | Keterangan |
| :--- | :--- | :--- |
| `HOST` | `0.0.0.0` | Host binding server Flask di dalam container |
| `PORT` | `5000` | Port aplikasi web |
| `WA_NOTIFICATION_ENABLED` | `True` | Mengaktifkan/menonaktifkan peringatan WhatsApp |
| `WA_METHOD` | `webhook` | Metode pengiriman notifikasi (`webhook` atau `fonnte`) |
| `WEBHOOK_URL` | - | URL endpoint bot WhatsApp |
| `WEBHOOK_SECRET` | - | Kunci otorisasi rahasia bot |
| `WA_TARGET_PHONE` | - | Nomor WhatsApp tujuan notifikasi |

> *Jika berkas `.env` tidak disediakan saat pengujian dosen/asisten lab, aplikasi tetap akan berjalan normal pada port 5000.*

---

## 4. Panduan Singkat Pengujian Fitur

1. **Pengujian Deteksi Akustik Kontinu:**
   * Buka dashboard di browser dan klik tombol **"Mulai Monitoring Kontinu"** (izinkan akses mikrofon browser).
   * Putar audio sampel gergaji mesin dekat mikrofon (jarak 10–30 cm).
   * Perhatikan indikator:
     * Gelombang osiloskop akan bereaksi secara real-time.
     * Bilah **Durasi Sawmill Kontinu** akan mulai menghitung (`0.5s -> 1.0s -> ... -> 10.0s`).
     * Jika suara sawmill bertahan terus selama **≥ 10 detik**, status notifikasi WhatsApp akan berubah menjadi `Terkirim` dan pesan peringatan otomatis dikirimkan.
2. **Pengujian Notifikasi Cepat:**
   * Klik tombol **"Tes Notif WA"** pada kontrol dashboard untuk memverifikasi jalur pengiriman pesan ke WhatsApp tanpa menunggu 10 detik.

---

## 5. Struktur Berkas Repositori

```text
├── Dockerfile              # Resep build container Linux & audio libraries
├── docker-compose.yml      # Konfigurasi container runner & port mapping 5000:5000
├── requirements.txt        # Daftar pustaka Python yang dibutuhkan
├── app.py                  # Aplikasi backend Flask, ring buffer & inferensi ML
├── model_terbaik_klasifikasi_audio.joblib # Model ML Random Forest (±58 KB)
├── templates/
│   └── index.html          # Frontend dashboard responsif (Anti-Slop UI)
├── .env.example            # Template konfigurasi environment yang aman
├── .gitignore              # Daftar berkas yang diabaikan Git (.env, cache, dll)
└── README.md               # Dokumentasi dan petunjuk menjalankan proyek
```

---

## 6. Informasi Kelompok & Rekap Deploy

* **Mata Kuliah:** Proyek Sains Data (Semester 5)
* **Kelompok:** [Isi Nama / Nomor Kelompok Anda]
* **Anggota:**
  1. [Nama Mahasiswa 1] - [NIM]
  2. [Nama Mahasiswa 2] - [NIM]
* **Tautan Repositori GitHub:** `https://github.com/<USERNAME>/<REPO_NAME>`
