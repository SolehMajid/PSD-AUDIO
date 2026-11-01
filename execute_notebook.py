import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import tempfile

os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

notebook_path = 'klasifikasi-tanpa-strategi.ipynb'
print(f"Membaca notebook: {notebook_path}")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

print("Mulai mengeksekusi semua cell di notebook...")
ep.preprocess(nb, {'metadata': {'path': '.'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Sukses! Seluruh cell di {notebook_path} telah selesai dieksekusi dan disimpan.")

