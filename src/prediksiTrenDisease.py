# Import library yang dibutuhkan
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load dataset
df = pd.read_csv('chronic_disease_children_trend.csv')

# Missing Value Handling

# Cek dan Imputasi Missing Values
if df.isnull().sum().sum() > 0:
    for col in ['Asthma_Prevalence_pct', 'Pneumonia_Prevalence_pct', 'Anemia_Prevalence_pct']:
        df[col].fillna(df[col].median(), inplace=True)

# Hapus Duplikasi Baris
if df.duplicated().sum() > 0:
    df.drop_duplicates(inplace=True)

# Encoding Kategori

# Encoding kategori pada kolom 'Province'
le = LabelEncoder()
df['Province_encoded'] = le.fit_transform(df['Province'])

# Tampilkan hasil encoding
print(df[['Province', 'Province_encoded']].drop_duplicates().sort_values(by='Province_encoded'))

# Persiapan untuk Tahap Normalisasi
features = ['Year', 'Province_encoded', 'Asthma_Prevalence_pct', 'Pneumonia_Prevalence_pct']
target = 'Anemia_Prevalence_pct'
X = df[features]
y = df[target]

