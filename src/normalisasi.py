import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

df = pd.read_csv('data/chronic_disease_children_trend.csv')

for col in ['Asthma_Prevalence_pct', 'Pneumonia_Prevalence_pct', 'Anemia_Prevalence_pct']:
    median_val = df[col].median()
    
    df[col] = df[col].fillna(median_val)

df.drop_duplicates(inplace=True)

le = LabelEncoder()
df['Province_encoded'] = le.fit_transform(df['Province'])

features = ['Year', 'Province_encoded', 'Asthma_Prevalence_pct', 'Pneumonia_Prevalence_pct']
target = 'Anemia_Prevalence_pct'
X = df[features] 
y = df[target]   

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)

y_reshaped = y.values.reshape(-1, 1)
y_scaled = scaler_y.fit_transform(y_reshaped)

X_scaled_df = pd.DataFrame(X_scaled, columns=features)
y_scaled_df = pd.DataFrame(y_scaled, columns=[target])

df_normalized = pd.concat([X_scaled_df, y_scaled_df], axis=1)

print("Data Setelah Normalisasi (Fitur dan Target):")
print(df_normalized.head())