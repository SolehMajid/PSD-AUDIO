# Rekomendasi Algoritma dan Strategi Optimasi Akurasi Klasifikasi Audio Sawmill dan Non-Sawmill

## 1. Rekomendasi Algoritma Machine Learning

Penelitian ini bertujuan untuk melakukan klasifikasi audio menjadi dua kelas, yaitu sawmill dan non-sawmill. Dataset yang digunakan merupakan hasil ekstraksi fitur audio menggunakan Librosa, yang terdiri dari MFCC, Chroma, Spectral Centroid, Spectral Bandwidth, Spectral Rolloff, Zero Crossing Rate (ZCR), dan Root Mean Square (RMS). Setiap fitur direpresentasikan menggunakan nilai mean dan standard deviation sehingga menghasilkan data numerik berbentuk tabular.

Berdasarkan karakteristik dataset tersebut, algoritma machine learning yang direkomendasikan adalah sebagai berikut.

### 1.1 XGBoost (Extreme Gradient Boosting) dan LightGBM

XGBoost dan LightGBM merupakan algoritma ensemble learning berbasis gradient boosting yang menggunakan kumpulan decision tree untuk membangun model klasifikasi.

Algoritma ini direkomendasikan karena mampu mempelajari hubungan non-linear serta interaksi antarfitur audio. Dalam klasifikasi sawmill dan non-sawmill, model dapat mempelajari kombinasi karakteristik MFCC, spectral centroid, spectral bandwidth, dan RMS yang membedakan suara mesin penggergajian kayu dari suara lingkungan lainnya.

Kelebihan:
- Mampu menangani data numerik tabular dengan banyak fitur.
- Mampu mempelajari hubungan non-linear antarfitur.
- Mendukung regularisasi untuk mengendalikan overfitting.
- Dapat digunakan untuk menganalisis feature importance.
- Tidak memerlukan normalisasi fitur menggunakan StandardScaler.

XGBoost direkomendasikan sebagai salah satu kandidat model utama, sedangkan LightGBM digunakan sebagai algoritma pembanding.

### 1.2 Random Forest Classifier

Random Forest merupakan algoritma ensemble learning yang menggabungkan prediksi dari sejumlah decision tree.

Algoritma ini direkomendasikan karena relatif stabil dalam menangani data numerik tabular serta mampu mempelajari hubungan non-linear antarfitur audio.

Dalam penelitian ini, Random Forest dapat digunakan untuk mengidentifikasi fitur yang memiliki kontribusi besar terhadap klasifikasi suara sawmill dan non-sawmill.

Kelebihan:
- Relatif mudah diterapkan.
- Tidak memerlukan normalisasi fitur.
- Dapat mengurangi varians melalui penggabungan banyak decision tree.
- Mendukung analisis feature importance.
- Dapat digunakan sebagai baseline pembanding algoritma boosting.

Random Forest digunakan sebagai model pembanding untuk mengetahui apakah pendekatan bagging mampu menghasilkan performa yang kompetitif dibandingkan gradient boosting dan SVM.

### 1.3 Support Vector Machine (SVM) dengan Kernel RBF

Support Vector Machine dengan Radial Basis Function (RBF) Kernel merupakan algoritma yang direkomendasikan untuk klasifikasi data dengan jumlah fitur relatif banyak.

SVM-RBF mampu membentuk batas keputusan non-linear sehingga dapat digunakan untuk memisahkan karakteristik suara sawmill dan non-sawmill yang memiliki pola kompleks.

Kelebihan:
- Efektif untuk klasifikasi biner.
- Cocok untuk dataset dengan jumlah fitur relatif banyak.
- Mampu membentuk batas klasifikasi non-linear.
- Memiliki mekanisme regularisasi melalui parameter C.
- Dapat memberikan performa yang baik pada dataset audio berukuran kecil hingga menengah.

SVM-RBF direkomendasikan sebagai kandidat model utama karena dataset penelitian sudah berbentuk fitur numerik hasil ekstraksi Librosa.

Namun, SVM sensitif terhadap skala fitur sehingga perlu menggunakan StandardScaler atau MinMaxScaler.

### 1.4 Rekomendasi Pemilihan Algoritma

Berdasarkan ketiga pendekatan tersebut, algoritma yang direkomendasikan untuk eksperimen penelitian adalah:

1. XGBoost sebagai kandidat model utama.
2. SVM-RBF sebagai kandidat model utama.
3. LightGBM sebagai model pembanding.
4. Random Forest sebagai baseline ensemble learning.

Pemilihan model terbaik dilakukan berdasarkan hasil validasi silang dan evaluasi pada data testing yang tidak digunakan selama proses pelatihan maupun tuning.

Tidak ada algoritma yang dapat dijamin menghasilkan akurasi mendekati 100% pada seluruh dataset. Oleh karena itu, pemilihan algoritma terbaik harus didasarkan pada hasil eksperimen.

---

## 2. Strategi Optimasi Akurasi Klasifikasi

Untuk memperoleh performa klasifikasi yang optimal, penelitian ini menggunakan beberapa strategi berikut.

### 2.1 Feature Scaling (Normalisasi Fitur)

Feature scaling digunakan untuk menyamakan skala fitur sehingga algoritma yang sensitif terhadap perbedaan rentang nilai dapat bekerja secara optimal.

Dalam penelitian ini, StandardScaler direkomendasikan untuk algoritma SVM-RBF.

StandardScaler melakukan transformasi terhadap setiap fitur sehingga memiliki rata-rata mendekati 0 dan standard deviation mendekati 1.

Penerapan scaling harus dilakukan menggunakan data training saja, kemudian transformasi yang sama diterapkan pada data validasi dan testing.

Untuk menghindari data leakage, proses scaling ditempatkan di dalam pipeline machine learning.

XGBoost, LightGBM, dan Random Forest tidak memerlukan StandardScaler secara wajib karena algoritma berbasis decision tree tidak sensitif terhadap skala fitur.

### 2.2 Stratified Cross-Validation

Penelitian ini menggunakan Stratified 5-Fold Cross-Validation untuk mengevaluasi kestabilan model.

Stratified Cross-Validation membagi data training menjadi lima bagian dengan mempertahankan proporsi kelas sawmill dan non-sawmill pada setiap fold.

Pada setiap iterasi:
- Empat fold digunakan untuk training.
- Satu fold digunakan untuk validasi.
- Proses diulang sebanyak lima kali.

Hasil evaluasi diperoleh dari rata-rata skor validasi seluruh fold.

Apabila dataset memiliki beberapa potongan audio yang berasal dari rekaman sumber yang sama, digunakan StratifiedGroupKFold atau GroupKFold untuk mencegah potongan audio dari rekaman yang sama masuk ke training dan validasi secara bersamaan.

Strategi ini penting untuk memastikan bahwa model tidak hanya mengenali karakteristik rekaman tertentu, tetapi mampu melakukan generalisasi terhadap rekaman baru.

### 2.3 Hyperparameter Tuning

Hyperparameter tuning dilakukan untuk memperoleh kombinasi parameter yang memberikan performa terbaik.

Parameter yang dapat dioptimalkan antara lain:

**SVM-RBF:**
- C
- Gamma

**XGBoost:**
- n_estimators
- max_depth
- learning_rate
- subsample
- colsample_bytree

**LightGBM:**
- n_estimators
- max_depth
- learning_rate
- num_leaves

**Random Forest:**
- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf

Proses optimasi dapat menggunakan GridSearchCV atau RandomizedSearchCV dengan Stratified 5-Fold Cross-Validation.

Pemilihan parameter dilakukan berdasarkan skor validasi, bukan berdasarkan akurasi training.

### 2.4 Penanganan Imbalanced Dataset

Distribusi kelas sawmill dan non-sawmill perlu diperiksa sebelum proses pelatihan.

Jika jumlah sampel kedua kelas relatif seimbang, SMOTE tidak perlu digunakan.

Jika terdapat ketidakseimbangan kelas, beberapa pendekatan yang dapat digunakan adalah:

- Class Weight: memberikan bobot lebih besar kepada kelas minoritas.
- SMOTE: menghasilkan sampel sintetis untuk kelas minoritas.
- Random Oversampling: meningkatkan jumlah sampel kelas minoritas dengan mengambil ulang sampel yang tersedia.

Apabila menggunakan SMOTE, proses oversampling hanya dilakukan pada data training di setiap fold validasi silang. SMOTE tidak boleh diterapkan sebelum pembagian training dan testing.

Untuk SVM dan Random Forest, class_weight='balanced' dapat digunakan sebagai alternatif penanganan imbalance.

Pada XGBoost dan LightGBM, penanganan imbalance dapat menggunakan parameter yang sesuai, seperti scale_pos_weight atau class_weight.

### 2.5 Pencegahan Data Leakage

Data leakage merupakan kondisi ketika informasi dari data testing atau data validasi secara tidak langsung digunakan selama proses pelatihan.

Dalam penelitian klasifikasi audio, data leakage dapat terjadi apabila beberapa potongan audio yang berasal dari rekaman sumber yang sama tersebar ke data training dan testing.

Untuk menghindari hal tersebut, pembagian dataset harus memperhatikan sumber rekaman.

Jika satu rekaman audio panjang dipotong menjadi beberapa sampel, seluruh potongan dari rekaman tersebut harus ditempatkan pada kelompok yang sama.

Strategi ini bertujuan agar evaluasi model mencerminkan kemampuan klasifikasi terhadap rekaman audio yang benar-benar baru.

### 2.6 Evaluasi Model

Evaluasi performa klasifikasi dilakukan menggunakan beberapa metrik:

- Accuracy: mengukur proporsi prediksi yang benar.
- Precision: mengukur ketepatan prediksi suatu kelas.
- Recall: mengukur kemampuan model mendeteksi sampel dari suatu kelas.
- F1-Score: menggabungkan precision dan recall.
- ROC-AUC: mengukur kemampuan model membedakan kedua kelas.
- Confusion Matrix: menunjukkan jumlah prediksi benar dan salah pada setiap kelas.

Pemilihan model terbaik tidak hanya didasarkan pada accuracy, tetapi juga mempertimbangkan precision, recall, dan F1-Score.

Apabila tujuan utama penelitian adalah mendeteksi keberadaan suara sawmill, recall pada kelas sawmill perlu diperhatikan agar model tidak terlalu banyak melewatkan suara sawmill.

---

## 3. Rancangan Eksperimen Penelitian

Rancangan eksperimen yang direkomendasikan adalah sebagai berikut:

| Eksperimen | Algoritma | Strategi |
|---|---|---|
| 1 | SVM-RBF | StandardScaler + Hyperparameter Tuning |
| 2 | Random Forest | Class Weight + Hyperparameter Tuning |
| 3 | XGBoost | Regularisasi + Hyperparameter Tuning |
| 4 | LightGBM | Regularisasi + Hyperparameter Tuning |
| 5 | Model terbaik | Evaluasi menggunakan data testing |

Seluruh algoritma menggunakan fitur hasil ekstraksi Librosa yang sama agar perbandingan performa bersifat adil.

Proses pelatihan menggunakan Stratified 5-Fold Cross-Validation. Apabila dataset terdiri dari beberapa potongan rekaman sumber yang sama, digunakan pemisahan berbasis grup.

Model terbaik ditentukan berdasarkan performa validasi silang, kemudian dievaluasi menggunakan data testing yang tidak pernah digunakan dalam proses training maupun tuning.

## 4. Kesimpulan

Berdasarkan karakteristik fitur audio yang digunakan, XGBoost, LightGBM, Random Forest, dan SVM-RBF merupakan algoritma yang layak digunakan dalam klasifikasi audio sawmill dan non-sawmill.

XGBoost dan SVM-RBF direkomendasikan sebagai kandidat model utama, sedangkan Random Forest dan LightGBM digunakan sebagai model pembanding.

Untuk meningkatkan performa klasifikasi, penelitian ini menerapkan feature scaling pada algoritma yang memerlukannya, Stratified Cross-Validation, hyperparameter tuning, penanganan imbalance dataset, serta pencegahan data leakage.

Kombinasi algoritma dan strategi tersebut diharapkan dapat menghasilkan model klasifikasi dengan performa yang optimal dan kemampuan generalisasi yang baik terhadap rekaman audio baru.

Namun, akurasi mendekati 100% tidak dapat dijamin hanya melalui pemilihan algoritma. Hasil tersebut harus dibuktikan melalui eksperimen yang menggunakan pembagian dataset dan evaluasi yang benar.
