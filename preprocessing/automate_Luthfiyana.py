import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download resource NLTK
# Menggunakan quiet agar log tidak penuh
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def clean_text(text):
    """
    Fungsi untuk membersihkan teks:
    - Lowercase
    - Hapus karakter non-huruf
    - Tokenisasi
    - Lemmatization & Hapus Stopwords
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Ubah ke lowercase
    text = str(text).lower()
    # Hapus karakter spesial & angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenisasi (split by spasi)
    words = text.split()
    # Hapus stopwords dan lakukan lemmatization
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Gabung kembali
    return " ".join(cleaned_words)

def process_data(input_path, output_path):
    print("Memulai Otomatisasi Preprocessing...")
    
    # 1. Load Dataset
    print(f"Memuat dataset dari: {input_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan di: {input_path}")
        
    df = pd.read_csv(input_path)
    
    # 2. Perbaikan Struktur Awal
    # Hapus kolom Unnamed jika ada
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    # Pastikan nama kolom
    if len(df.columns) >= 2:
        df.columns = ['label', 'text']
    
    # Ubah tipe data label menjadi integer
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label']) # Hapus baris jika label NaN
    df['label'] = df['label'].astype(int)
    
    print(f"Ukuran awal dataset: {df.shape}")

    # 3. Hapus Duplikasi
    initial_rows = df.shape[0]
    df = df.drop_duplicates(keep='first')
    print(f"Duplikasi dihapus: {initial_rows - df.shape[0]} baris.")
    
    # 4. Cleaning Teks
    print("Sedang melakukan cleaning teks (ini mungkin memakan waktu)...")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Hapus baris yang kosong setelah cleaning (jika ada)
    df = df[df['clean_text'].str.strip() != '']
    
    # 5. Finalisasi Kolom
    # Rename kolom 'label' menjadi 'target'
    df = df.rename(columns={'label': 'target'})
    
    # Hanya ambil kolom hasil
    final_df = df[['clean_text', 'target']]
    
    # 6. Simpan Data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # compression='zip'
    final_df.to_csv(output_path, index=False, compression='zip')
    
    print(f"Preprocessing Selesai! Data bersih disimpan di: {output_path}")
    print(f"Ukuran akhir dataset: {final_df.shape}")
    print("Contoh data:")
    print(final_df.head())

if __name__ == "__main__":
    # Pengaturan Path
    INPUT_FILE = os.path.join("..", "email_spam_raw", "spam.zip")
    
    # Output: Disimpan di folder spam_cleaned yang sejajar dengan script ini
    OUTPUT_FILE = os.path.join("spam_cleaned", "spam_cleaned.zip")
    
    try:
        process_data(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")