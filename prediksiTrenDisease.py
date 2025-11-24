# Import library yang dibutuhkan
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('data/chronic_disease_children_trend.csv')

# Encoding kategori pada kolom 'Province'
le = LabelEncoder()
df['Province_encoded'] = le.fit_transform(df['Province'])

# Tampilkan hasil encoding
print(df[['Province', 'Province_encoded']].drop_duplicates())
