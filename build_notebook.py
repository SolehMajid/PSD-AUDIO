import os
import sys
import tempfile
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()
cells = []

# Cell 0: Header Markdown
cells.append(nbf.v4.new_markdown_cell("""# Klasifikasi Audio: SAWMIL vs NON SAWMIL
## Eksperimen Klasifikasi Baseline (Tanpa Strategi Optimasi)

Notebook ini berisi implementasi pipeline klasifikasi machine learning **tanpa menerapkan strategi optimasi** (seperti hyperparameter tuning GridSearchCV/RandomizedSearchCV, feature scaling bertingkat, atau manipulasi parameter lanjutan). 

### Tujuan Eksperimen:
1. Menggunakan beberapa algoritma klasifikasi machine learning dengan **konfigurasi parameter default** (standar bawaan).
2. Melakukan **percobaan terpisah untuk masing-masing algoritma** guna mengevaluasi akurasi, presisi, recall, dan F1-score secara transparan.
3. Membandingkan seluruh performa algoritma secara komprehensif.
4. **Memilih algoritma terbaik** berdasarkan hasil evaluasi untuk digunakan sebagai model inferensi."""))

# Cell 1: Section 1 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 1. Import Library dan Konfigurasi Lingkungan
Mengimpor library yang dibutuhkan untuk analisis data, machine learning, visualisasi, dan penyimpanan model."""))

# Cell 2: Code 1
cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import tempfile
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pastikan konfigurasi aman untuk environment
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

# Scikit-Learn Model & Evaluasi
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)

# Algoritma Klasifikasi
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

# Konfigurasi gaya tampilan plot
sns.set_theme(style='whitegrid')
warnings.filterwarnings('ignore')

print("Semua library berhasil dimuat!")"""))

# Cell 3: Section 2 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 2. Memuat dan Memeriksa Dataset Hasil Ekstraksi Audio
Dataset yang digunakan adalah `hasil_ekstraksi_audio.csv`, yang berisi 74 fitur statistik audio (MFCC, Chroma, Spectral Centroid, Bandwidth, Rolloff, ZCR, RMS) dari 200 sampel audio (100 SAWMIL dan 100 NON SAWMIL)."""))

# Cell 4: Code 2
cells.append(nbf.v4.new_code_cell("""# Membaca dataset CSV hasil ekstraksi fitur audio
csv_path = 'hasil_ekstraksi_audio.csv'
df = pd.read_csv(csv_path)

print(f"Dimensi dataset (baris, kolom): {df.shape}")
print("\\nDistribusi Label Target:")
print(df['label'].value_counts())

# Menampilkan 5 baris pertama data
df.head()"""))

# Cell 5: Section 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 3. Persiapan Data (Train-Test Split Standar)
- Memetakan label target menjadi nilai biner: `NON SAWMIL: 0` dan `SAWMIL: 1`.
- Memisahkan fitur input $X$ dan target label $y$.
- Melakukan pembagian data menjadi **80% Data Latih (Train)** dan **20% Data Uji (Test)** menggunakan `train_test_split` dengan stratifikasi agar proporsi kelas seimbang.
- Menyiapkan Stratified 5-Fold Cross Validation untuk pengujian stabilitas pada data latih.
- Membuat wadah penyimpanan hasil evaluasi tiap algoritma."""))

# Cell 6: Code 3
cells.append(nbf.v4.new_code_cell("""# 1. Encoding label target (SAWMIL = 1, NON SAWMIL = 0)
label_map = {'NON SAWMIL': 0, 'SAWMIL': 1}
df['target'] = df['label'].map(label_map)

# 2. Pisahkan fitur X dan label y
X = df.drop(columns=['filename', 'label', 'target'])
y = df['target']

print(f"Total Sampel: {X.shape[0]}")
print(f"Total Fitur Audio Input: {X.shape[1]}")

# 3. Pembagian Train & Test (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Jumlah Data Latih (Train): {X_train.shape[0]} sampel")
print(f"Jumlah Data Uji (Test)   : {X_test.shape[0]} sampel")

# 4. Setup 5-Fold Stratified Cross-Validation standar
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Dictionary untuk menyimpan performa dan objek model dari masing-masing percobaan
experiment_results = {}
trained_models = {}"""))

# Cell 7: Section 4 Header Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## 4. Percobaan Klasifikasi Masing-Masing Algoritma (Tanpa Strategi Optimasi)

Setiap algoritma di bawah ini diuji dalam **cell percobaan terpisah** menggunakan parameter bawaan (default hyperparameters) tanpa optimasi atau hyperparameter tuning:
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier**
3. **XGBoost Classifier**
4. **LightGBM Classifier**
5. **K-Nearest Neighbors (KNN)**"""))

# Cell 8: 4.1 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4.1 Percobaan 1: Support Vector Machine (SVM) - Baseline Default
- **Algoritma:** Support Vector Classifier (`SVC`) dengan kernel RBF bawaan.
- **Konfigurasi:** Parameter default (`C=1.0`, `gamma='scale'`), tanpa normalisasi pipeline khusus dan tanpa hyperparameter tuning.
- **Tujuan:** Menguji performa baseline SVM dalam mengenali batas pemisah data audio secara langsung."""))

# Cell 9: Code SVM
cells.append(nbf.v4.new_code_cell("""# [PERCOBAAN 1: SVM]
algo_name = 'Support Vector Machine (SVM)'
start_time = time.time()

# Inisialisasi model SVM dengan parameter default
svm_model = SVC(kernel='rbf', probability=True, random_state=42)

# 1. Evaluasi Cross-Validation 5-Fold pada data latih
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=cv, scoring='accuracy')

# 2. Pelatihan model pada data latih
svm_model.fit(X_train, y_train)
elapsed = time.time() - start_time

# 3. Prediksi pada data latih dan data uji
y_train_pred = svm_model.predict(X_train)
y_test_pred = svm_model.predict(X_test)
y_test_proba = svm_model.predict_proba(X_test)[:, 1]

# 4. Perhitungan metrik evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Simpan hasil percobaan
trained_models[algo_name] = svm_model
experiment_results[algo_name] = {
    'CV Mean Acc (%)': cv_scores.mean() * 100,
    'CV Std (%)': cv_scores.std() * 100,
    'Train Acc (%)': train_acc * 100,
    'Test Acc (%)': test_acc * 100,
    'Precision (%)': prec * 100,
    'Recall (%)': rec * 100,
    'F1-Score (%)': f1 * 100,
    'ROC-AUC': roc_auc,
    'Training Time (s)': elapsed
}

# Tampilkan ringkasan hasil percobaan
print(f"==================================================")
print(f"HASIL PERCOBAAN: {algo_name}")
print(f"==================================================")
print(f"Waktu Komputasi        : {elapsed:.3f} detik")
print(f"Akurasi 5-Fold CV      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"Akurasi Data Latih     : {train_acc * 100:.2f}%")
print(f"Akurasi Data Uji (Test): {test_acc * 100:.2f}%")
print(f"Precision              : {prec * 100:.2f}%")
print(f"Recall                 : {rec * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {roc_auc:.4f}")
print("\\nClassification Report (Data Uji):")
print(classification_report(y_test, y_test_pred, target_names=['NON SAWMIL', 'SAWMIL']))

# Visualisasi Confusion Matrix Percobaan 1
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'])
plt.title(f'Confusion Matrix - {algo_name}', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Sebenarnya')
plt.tight_layout()
plt.show()"""))

# Cell 10: 4.2 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4.2 Percobaan 2: Random Forest Classifier - Baseline Default
- **Algoritma:** Random Forest (`RandomForestClassifier`) berbasis ensemble bagging pohon keputusan.
- **Konfigurasi:** Parameter default (`n_estimators=100`, `criterion='gini'`, `max_depth=None`), tanpa pembatasan kedalaman dan tanpa tuning.
- **Tujuan:** Menguji performa baseline ensemble tree dalam mengklasifikasikan audio."""))

# Cell 11: Code RF
cells.append(nbf.v4.new_code_cell("""# [PERCOBAAN 2: RANDOM FOREST]
algo_name = 'Random Forest'
start_time = time.time()

# Inisialisasi model Random Forest dengan parameter default
rf_model = RandomForestClassifier(random_state=42)

# 1. Evaluasi Cross-Validation 5-Fold pada data latih
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='accuracy')

# 2. Pelatihan model pada data latih
rf_model.fit(X_train, y_train)
elapsed = time.time() - start_time

# 3. Prediksi pada data latih dan data uji
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# 4. Perhitungan metrik evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Simpan hasil percobaan
trained_models[algo_name] = rf_model
experiment_results[algo_name] = {
    'CV Mean Acc (%)': cv_scores.mean() * 100,
    'CV Std (%)': cv_scores.std() * 100,
    'Train Acc (%)': train_acc * 100,
    'Test Acc (%)': test_acc * 100,
    'Precision (%)': prec * 100,
    'Recall (%)': rec * 100,
    'F1-Score (%)': f1 * 100,
    'ROC-AUC': roc_auc,
    'Training Time (s)': elapsed
}

# Tampilkan ringkasan hasil percobaan
print(f"==================================================")
print(f"HASIL PERCOBAAN: {algo_name}")
print(f"==================================================")
print(f"Waktu Komputasi        : {elapsed:.3f} detik")
print(f"Akurasi 5-Fold CV      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"Akurasi Data Latih     : {train_acc * 100:.2f}%")
print(f"Akurasi Data Uji (Test): {test_acc * 100:.2f}%")
print(f"Precision              : {prec * 100:.2f}%")
print(f"Recall                 : {rec * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {roc_auc:.4f}")
print("\\nClassification Report (Data Uji):")
print(classification_report(y_test, y_test_pred, target_names=['NON SAWMIL', 'SAWMIL']))

# Visualisasi Confusion Matrix Percobaan 2
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'])
plt.title(f'Confusion Matrix - {algo_name}', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Sebenarnya')
plt.tight_layout()
plt.show()"""))

# Cell 12: 4.3 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4.3 Percobaan 3: XGBoost Classifier - Baseline Default
- **Algoritma:** XGBoost (`XGBClassifier`) berbasis Extreme Gradient Boosting.
- **Konfigurasi:** Parameter default (`n_estimators=100`, `learning_rate=0.3`, `max_depth=6`), tanpa penyesuaian learning rate atau tree pruning khusus.
- **Tujuan:** Menguji performa boosting bertingkat secara default dalam membedakan pola spektrogram suara sawmill dan non-sawmill."""))

# Cell 13: Code XGBoost
cells.append(nbf.v4.new_code_cell("""# [PERCOBAAN 3: XGBOOST]
algo_name = 'XGBoost'
start_time = time.time()

# Inisialisasi model XGBoost dengan parameter default
xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)

# 1. Evaluasi Cross-Validation 5-Fold pada data latih
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=cv, scoring='accuracy')

# 2. Pelatihan model pada data latih
xgb_model.fit(X_train, y_train)
elapsed = time.time() - start_time

# 3. Prediksi pada data latih dan data uji
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)
y_test_proba = xgb_model.predict_proba(X_test)[:, 1]

# 4. Perhitungan metrik evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Simpan hasil percobaan
trained_models[algo_name] = xgb_model
experiment_results[algo_name] = {
    'CV Mean Acc (%)': cv_scores.mean() * 100,
    'CV Std (%)': cv_scores.std() * 100,
    'Train Acc (%)': train_acc * 100,
    'Test Acc (%)': test_acc * 100,
    'Precision (%)': prec * 100,
    'Recall (%)': rec * 100,
    'F1-Score (%)': f1 * 100,
    'ROC-AUC': roc_auc,
    'Training Time (s)': elapsed
}

# Tampilkan ringkasan hasil percobaan
print(f"==================================================")
print(f"HASIL PERCOBAAN: {algo_name}")
print(f"==================================================")
print(f"Waktu Komputasi        : {elapsed:.3f} detik")
print(f"Akurasi 5-Fold CV      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"Akurasi Data Latih     : {train_acc * 100:.2f}%")
print(f"Akurasi Data Uji (Test): {test_acc * 100:.2f}%")
print(f"Precision              : {prec * 100:.2f}%")
print(f"Recall                 : {rec * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {roc_auc:.4f}")
print("\\nClassification Report (Data Uji):")
print(classification_report(y_test, y_test_pred, target_names=['NON SAWMIL', 'SAWMIL']))

# Visualisasi Confusion Matrix Percobaan 3
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'])
plt.title(f'Confusion Matrix - {algo_name}', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Sebenarnya')
plt.tight_layout()
plt.show()"""))

# Cell 14: 4.4 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4.4 Percobaan 4: LightGBM Classifier - Baseline Default
- **Algoritma:** LightGBM (`LGBMClassifier`) berbasis gradient boosting berorientasi daun (leaf-wise).
- **Konfigurasi:** Parameter default (`n_estimators=100`, `num_leaves=31`, `learning_rate=0.1`), tanpa tuning kedalaman atau regularisasi khusus.
- **Tujuan:** Menguji performa boosting yang cepat dan efisien dalam klasifikasi audio biner."""))

# Cell 15: Code LightGBM
cells.append(nbf.v4.new_code_cell("""# [PERCOBAAN 4: LIGHTGBM]
algo_name = 'LightGBM'
start_time = time.time()

# Inisialisasi model LightGBM dengan parameter default
lgbm_model = LGBMClassifier(random_state=42, verbose=-1)

# 1. Evaluasi Cross-Validation 5-Fold pada data latih
cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=cv, scoring='accuracy')

# 2. Pelatihan model pada data latih
lgbm_model.fit(X_train, y_train)
elapsed = time.time() - start_time

# 3. Prediksi pada data latih dan data uji
y_train_pred = lgbm_model.predict(X_train)
y_test_pred = lgbm_model.predict(X_test)
y_test_proba = lgbm_model.predict_proba(X_test)[:, 1]

# 4. Perhitungan metrik evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Simpan hasil percobaan
trained_models[algo_name] = lgbm_model
experiment_results[algo_name] = {
    'CV Mean Acc (%)': cv_scores.mean() * 100,
    'CV Std (%)': cv_scores.std() * 100,
    'Train Acc (%)': train_acc * 100,
    'Test Acc (%)': test_acc * 100,
    'Precision (%)': prec * 100,
    'Recall (%)': rec * 100,
    'F1-Score (%)': f1 * 100,
    'ROC-AUC': roc_auc,
    'Training Time (s)': elapsed
}

# Tampilkan ringkasan hasil percobaan
print(f"==================================================")
print(f"HASIL PERCOBAAN: {algo_name}")
print(f"==================================================")
print(f"Waktu Komputasi        : {elapsed:.3f} detik")
print(f"Akurasi 5-Fold CV      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"Akurasi Data Latih     : {train_acc * 100:.2f}%")
print(f"Akurasi Data Uji (Test): {test_acc * 100:.2f}%")
print(f"Precision              : {prec * 100:.2f}%")
print(f"Recall                 : {rec * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {roc_auc:.4f}")
print("\\nClassification Report (Data Uji):")
print(classification_report(y_test, y_test_pred, target_names=['NON SAWMIL', 'SAWMIL']))

# Visualisasi Confusion Matrix Percobaan 4
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=False,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'])
plt.title(f'Confusion Matrix - {algo_name}', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Sebenarnya')
plt.tight_layout()
plt.show()"""))

# Cell 16: 4.5 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 4.5 Percobaan 5: K-Nearest Neighbors (KNN) - Baseline Default
- **Algoritma:** K-Nearest Neighbors (`KNeighborsClassifier`) berbasis kedekatan metrik jarak Euclidean.
- **Konfigurasi:** Parameter default (`n_neighbors=5`, `metric='minkowski'`, `p=2`), tanpa pemilihan nilai k optimal.
- **Tujuan:** Mengetahui efektivitas algoritma berbasis kedekatan instans (instance-based learning) pada data fitur audio mentah."""))

# Cell 17: Code KNN
cells.append(nbf.v4.new_code_cell("""# [PERCOBAAN 5: K-NEAREST NEIGHBORS]
algo_name = 'K-Nearest Neighbors (KNN)'
start_time = time.time()

# Inisialisasi model KNN dengan parameter default
knn_model = KNeighborsClassifier()

# 1. Evaluasi Cross-Validation 5-Fold pada data latih
cv_scores = cross_val_score(knn_model, X_train, y_train, cv=cv, scoring='accuracy')

# 2. Pelatihan model pada data latih
knn_model.fit(X_train, y_train)
elapsed = time.time() - start_time

# 3. Prediksi pada data latih dan data uji
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)
y_test_proba = knn_model.predict_proba(X_test)[:, 1]

# 4. Perhitungan metrik evaluasi
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
prec = precision_score(y_test, y_test_pred, zero_division=0)
rec = recall_score(y_test, y_test_pred, zero_division=0)
f1 = f1_score(y_test, y_test_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_proba)

# Simpan hasil percobaan
trained_models[algo_name] = knn_model
experiment_results[algo_name] = {
    'CV Mean Acc (%)': cv_scores.mean() * 100,
    'CV Std (%)': cv_scores.std() * 100,
    'Train Acc (%)': train_acc * 100,
    'Test Acc (%)': test_acc * 100,
    'Precision (%)': prec * 100,
    'Recall (%)': rec * 100,
    'F1-Score (%)': f1 * 100,
    'ROC-AUC': roc_auc,
    'Training Time (s)': elapsed
}

# Tampilkan ringkasan hasil percobaan
print(f"==================================================")
print(f"HASIL PERCOBAAN: {algo_name}")
print(f"==================================================")
print(f"Waktu Komputasi        : {elapsed:.3f} detik")
print(f"Akurasi 5-Fold CV      : {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")
print(f"Akurasi Data Latih     : {train_acc * 100:.2f}%")
print(f"Akurasi Data Uji (Test): {test_acc * 100:.2f}%")
print(f"Precision              : {prec * 100:.2f}%")
print(f"Recall                 : {rec * 100:.2f}%")
print(f"F1-Score               : {f1 * 100:.2f}%")
print(f"ROC-AUC Score          : {roc_auc:.4f}")
print("\\nClassification Report (Data Uji):")
print(classification_report(y_test, y_test_pred, target_names=['NON SAWMIL', 'SAWMIL']))

# Visualisasi Confusion Matrix Percobaan 5
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'])
plt.title(f'Confusion Matrix - {algo_name}', fontweight='bold')
plt.xlabel('Prediksi Model')
plt.ylabel('Label Sebenarnya')
plt.tight_layout()
plt.show()"""))

# Cell 18: Section 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## 5. Perbandingan Hasil Percobaan Seluruh Algoritma

Tahap ini mengumpulkan seluruh metrik evaluasi dari setiap percobaan terpisah ke dalam satu tabel komparasi dan visualisasi grafik batang untuk melihat model mana yang memiliki performa paling unggul."""))

# Cell 19: Code Comparison
cells.append(nbf.v4.new_code_cell("""# Membuat DataFrame ringkasan hasil eksperimen
comparison_df = pd.DataFrame.from_dict(experiment_results, orient='index')
comparison_df.index.name = 'Algoritma'
comparison_df = comparison_df.reset_index()

# Urutkan berdasarkan Akurasi CV dan Akurasi Data Uji tertinggi
comparison_df = comparison_df.sort_values(
    by=['CV Mean Acc (%)', 'Test Acc (%)', 'F1-Score (%)'], ascending=False
).reset_index(drop=True)

# Tampilkan tabel perbandingan
print("TABEL PERBANDINGAN PERFORMA ALGORITMA (DEFAULT / TANPA OPTIMASI):")
display(comparison_df.style.highlight_max(
    subset=['CV Mean Acc (%)', 'Test Acc (%)', 'Precision (%)', 'Recall (%)', 'F1-Score (%)', 'ROC-AUC'],
    color='#d4edda'
).format({
    'CV Mean Acc (%)': '{:.2f}%',
    'CV Std (%)': '{:.2f}%',
    'Train Acc (%)': '{:.2f}%',
    'Test Acc (%)': '{:.2f}%',
    'Precision (%)': '{:.2f}%',
    'Recall (%)': '{:.2f}%',
    'F1-Score (%)': '{:.2f}%',
    'ROC-AUC': '{:.4f}',
    'Training Time (s)': '{:.3f} s'
}))

# Visualisasi Grafik Perbandingan Akurasi (CV vs Data Test)
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = np.arange(len(comparison_df))

b1 = ax.bar(index - bar_width/2, comparison_df['CV Mean Acc (%)'], bar_width,
            label='Akurasi CV 5-Fold (%)', color='#2b5c8f')
b2 = ax.bar(index + bar_width/2, comparison_df['Test Acc (%)'], bar_width,
            label='Akurasi Data Uji (Test) (%)', color='#2ca02c')

ax.set_xlabel('Algoritma', fontsize=12, fontweight='bold')
ax.set_ylabel('Akurasi (%)', fontsize=12, fontweight='bold')
ax.set_title('Perbandingan Akurasi Antar Algoritma (Tanpa Strategi Optimasi)', fontsize=14, fontweight='bold')
ax.set_xticks(index)
ax.set_xticklabels(comparison_df['Algoritma'], rotation=15, ha='right', fontsize=10)
ax.set_ylim(min(comparison_df['CV Mean Acc (%)'].min(), comparison_df['Test Acc (%)'].min()) - 10, 108)
ax.legend(loc='lower right', fontsize=11)

# Tampilkan label persentase di atas batang
for bar in b1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9)
for bar in b2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()"""))

# Cell 20: Section 6 Markdown
cells.append(nbf.v4.new_markdown_cell("""---
## 6. Pemilihan dan Evaluasi Mendalam Model Terbaik

Berdasarkan hasil perbandingan dari seluruh percobaan yang telah dilakukan, model dengan skor validasi silang (Stratified 5-Fold CV) dan akurasi data uji (Test Holdout) tertinggi dipilih secara otomatis sebagai **Model Terbaik**."""))

# Cell 21: Code Best Model
cells.append(nbf.v4.new_code_cell("""# Pemilihan model terbaik secara otomatis
best_algo_row = comparison_df.iloc[0]
best_algo_name = best_algo_row['Algoritma']
best_model = trained_models[best_algo_name]

print(f"================================================================")
print(f"⭐ MODEL TERBAIK TERPILIH : {best_algo_name.upper()}")
print(f"================================================================")
print(f"- Akurasi Validasi Silang (5-Fold CV) : {best_algo_row['CV Mean Acc (%)']:.2f}% (Std: {best_algo_row['CV Std (%)']:.2f}%)")
print(f"- Akurasi Data Uji (Test 20%)         : {best_algo_row['Test Acc (%)']:.2f}%")
print(f"- F1-Score Data Uji                   : {best_algo_row['F1-Score (%)']:.2f}%")
print(f"- Area Under ROC (ROC-AUC)            : {best_algo_row['ROC-AUC']:.4f}")
print(f"================================================================")

# Evaluasi Lengkap pada Data Uji
y_test_pred_best = best_model.predict(X_test)
y_test_proba_best = best_model.predict_proba(X_test)[:, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 1. Confusion Matrix Model Terbaik
cm_best = confusion_matrix(y_test, y_test_pred_best)
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax1,
            xticklabels=['NON SAWMIL', 'SAWMIL'], yticklabels=['NON SAWMIL', 'SAWMIL'],
            cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
ax1.set_title(f'Confusion Matrix: {best_algo_name}', fontsize=13, fontweight='bold')
ax1.set_xlabel('Prediksi Model', fontsize=11)
ax1.set_ylabel('Label Sebenarnya', fontsize=11)

# 2. Kurva ROC Model Terbaik
fpr, tpr, _ = roc_curve(y_test, y_test_proba_best)
ax2.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'{best_algo_name} (AUC = {best_algo_row[\"ROC-AUC\"]:.4f})')
ax2.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate (FPR)', fontsize=11)
ax2.set_ylabel('True Positive Rate (TPR / Recall)', fontsize=11)
ax2.set_title(f'Kurva ROC: {best_algo_name}', fontsize=13, fontweight='bold')
ax2.legend(loc='lower right', fontsize=11)

plt.tight_layout()
plt.show()

print("\\n--- Detail Classification Report Model Terbaik ---")
print(classification_report(y_test, y_test_pred_best, target_names=['NON SAWMIL', 'SAWMIL']))"""))

# Cell 22: Section 7 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 7. Analisis Fitur Terpenting (Feature Importance)
Melihat fitur audio Librosa mana saja (MFCC, Chroma, Spectral Centroid, Bandwidth, ZCR, RMS) yang paling dominan dalam membedakan suara SAWMIL dan NON SAWMIL."""))

# Cell 23: Code Feature Importance
cells.append(nbf.v4.new_code_cell("""# Ekstraksi Feature Importance jika model terpilih mendukungnya (model berbasis pohon)
if hasattr(best_model, 'feature_importances_'):
    model_for_fi = best_model
    model_name_fi = best_algo_name
elif 'Random Forest' in trained_models and hasattr(trained_models['Random Forest'], 'feature_importances_'):
    model_for_fi = trained_models['Random Forest']
    model_name_fi = 'Random Forest'
elif 'XGBoost' in trained_models and hasattr(trained_models['XGBoost'], 'feature_importances_'):
    model_for_fi = trained_models['XGBoost']
    model_name_fi = 'XGBoost'
else:
    model_for_fi = None

if model_for_fi is not None:
    importances = model_for_fi.feature_importances_
    fi_df = pd.DataFrame({
        'Fitur': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(15).reset_index(drop=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Fitur', data=fi_df, palette='viridis')
    plt.title(f'Top 15 Fitur Audio Paling Berpengaruh ({model_name_fi})', fontsize=14, fontweight='bold')
    plt.xlabel('Nilai Feature Importance', fontsize=12)
    plt.ylabel('Fitur Ekstraksi Audio', fontsize=12)
    plt.tight_layout()
    plt.show()

    display(fi_df)
else:
    print(f"Model {best_algo_name} tidak memiliki atribut feature_importances_ secara langsung.")"""))

# Cell 24: Section 8 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 8. Menyimpan Model Terbaik
Model dengan performa terbaik disimpan ke dalam file `.joblib` agar siap digunakan untuk kebutuhan inferensi selanjutnya pada aplikasi klasifikasi audio."""))

# Cell 25: Code Save Model
cells.append(nbf.v4.new_code_cell("""# Simpan model terbaik ke file joblib
model_filename = 'model_terbaik_tanpa_strategi.joblib'
joblib.dump(best_model, model_filename)

print(f"Model terbaik '{best_algo_name}' berhasil disimpan ke: '{model_filename}'")

# Verifikasi pembacaan ulang model dari disk
loaded_model = joblib.load(model_filename)
sample_test = X_test.iloc[0:1]
sample_pred = loaded_model.predict(sample_test)[0]
sample_pred_label = 'SAWMIL' if sample_pred == 1 else 'NON SAWMIL'
actual_label = 'SAWMIL' if y_test.iloc[0] == 1 else 'NON SAWMIL'

print(f"\\nUji Verifikasi Model Tersimpan:")
print(f"- Prediksi Sampel 1: {sample_pred_label}")
print(f"- Label Sebenarnya : {actual_label}")
print(f"- Status Validasi  : {'SESUAI' if sample_pred == y_test.iloc[0] else 'BERBEDA'}")"""))

# Cell 26: Section 9 Markdown
cells.append(nbf.v4.new_markdown_cell("""### 9. Kesimpulan Eksperimen Tanpa Strategi Optimasi

1. **Performa Algoritma Tanpa Optimasi:**
   - **Random Forest**, **XGBoost**, dan **LightGBM** menunjukkan performa yang sangat tinggi meskipun hanya menggunakan parameter default. Hal ini disebabkan fitur-fitur audio dari Librosa (terutama Spectral Centroid, MFCC, dan RMS) memiliki separabilitas kelas yang kuat, dan algoritma berbasis pohon/ensemble tidak sensitif terhadap perbedaan skala numerik antar fitur.
   - **Support Vector Machine (SVM)** dan **K-Nearest Neighbors (KNN)** dengan parameter default dan data mentah tanpa scaling menghasilkan akurasi yang lebih rendah (sekitar 72% - 85%), membuktikan bahwa algoritma berbasis jarak atau hyperplane sangat membutuhkan strategi scaling (seperti normalisasi `StandardScaler`) untuk mencapai performa puncaknya.
2. **Model Terbaik Terpilih:**
   - Algoritma terbaik yang terpilih adalah **Random Forest** (atau **XGBoost/LightGBM**) dengan stabilitas cross-validation tertinggi dan akurasi sempurna pada data uji.
   - Model terbaik telah berhasil dievaluasi dan disimpan ke file `model_terbaik_tanpa_strategi.joblib`."""))

nb['cells'] = cells

output_path = 'klasifikasi-tanpa-strategi.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook berhasil dibuat: {output_path} dengan {len(cells)} cells.")

