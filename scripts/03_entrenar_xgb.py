"""
ENTRENAMIENTO DE XGBOOST - Con ground truth
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_xgb.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_xgb.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder_xgb.pkl')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("=" * 70)
print("ENTRENAMIENTO XGBOOST")
print("=" * 70)

print(f"\n📂 Cargando datos desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ Datos cargados: {len(df)} filas")

features = [
    'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
    'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
    'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
    'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
]

for col in features:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=features)

if 'estado_real' not in df.columns:
    print("❌ No se encontró 'estado_real'")
    sys.exit(1)

X = df[features]
y = df['estado_real']
y = y.str.strip() if hasattr(y, 'str') else y

print(f"\n📊 Distribución de clases:")
print(y.value_counts())

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.3, random_state=42, stratify=y_enc)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n🚀 Entrenando XGBoost...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.2%}")
print("\n📋 Reporte:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

model_data = {'model': model, 'scaler': scaler, 'label_encoder': le, 'features': features, 'is_trained': True}
joblib.dump(model_data, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(le, ENCODER_PATH)

print(f"\n✅ Modelo guardado en: {MODEL_PATH}")