"""
ENTRENAMIENTO DE RANDOM FOREST
USA SOLO LAS 8 VARIABLES NUMÉRICAS DE DIAGNÓSTICO
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Dataset valvulas de control TY - Comparativa.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_rf.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_rf.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder_rf.pkl')

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

print("=" * 70)
print("ENTRENAMIENTO RANDOM FOREST - SOLO 8 VARIABLES")
print("=" * 70)

# 1. Cargar datos
print(f"\n📂 Cargando datos desde: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')
except:
    df = pd.read_csv(DATA_PATH, sep=',', encoding='utf-8')

print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

# 2. SOLO estas 8 variables (las que necesitamos para diagnóstico)
features = [
    'FRICCION RECOMENDADA', 
    'FRICCION MEDIDA',
    'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 
    'TORQUE - CARGA EN EL ASIENTO MEDIDA',
    'BANDA DE ERROR DINAMICA RECOMENDADA', 
    'BANDA DE ERROR DINAMICA MEDIDA',
    'LINEALIDAD DINAMICA RECOMENDADA', 
    'LINEALIDAD DINAMICA MEDIDA'
]

print(f"\n📊 Usando SOLO estas 8 variables:")
for f in features:
    print(f"   - {f}")

# 3. Buscar columna de diagnóstico experto
columna_target = None
posibles = ['DIAGNOSTICO_EXPERTO', 'DIAGNOSTICO', 'ESTADO', 'EXPERTO', 'CLASIFICACION']

for col in posibles:
    if col in df.columns:
        columna_target = col
        break

if columna_target:
    print(f"\n✅ Usando diagnóstico experto: '{columna_target}'")
else:
    print(f"\n⚠️ No se encontró columna de diagnóstico experto")
    print("   Se usará cálculo automático de estados")

# 4. Convertir SOLO las features numéricas
print("\n📝 Convirtiendo datos a formato numérico...")

for col in features:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')
    else:
        print(f"   ❌ Columna faltante: {col}")

# 5. Eliminar filas con valores nulos en las features
df_clean = df.dropna(subset=features)
print(f"✅ Filas después de limpieza: {len(df_clean)}")

# 6. Preparar X (solo las 8 features)
X = df_clean[features]

# 7. Preparar Y (diagnóstico)
if columna_target and columna_target in df_clean.columns:
    y = df_clean[columna_target]
    # Limpiar espacios en blanco en los nombres de clases
    y = y.str.strip() if hasattr(y, 'str') else y
    print(f"\n📊 Variable objetivo: {columna_target}")
    print(f"   Clases encontradas: {y.unique().tolist()}")
    print(f"   Distribución:")
    print(y.value_counts())
else:
    # Si no hay diagnóstico experto, calcular automático (solo como respaldo)
    print("\n📊 Calculando estados automáticamente...")
    
    def calcular_estado(row):
        desv_f = abs(row['FRICCION MEDIDA'] - row['FRICCION RECOMENDADA'])
        p_f = 1 if desv_f <= 5 else 2 if desv_f <= 10 else 3
        
        if row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] > 0:
            desv_c = abs((row['TORQUE - CARGA EN EL ASIENTO MEDIDA'] / row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] - 1) * 100)
        else:
            desv_c = 0
        p_c = 1 if desv_c <= 10 else 2 if desv_c <= 25 else 3
        
        p_b = 1 if row['BANDA DE ERROR DINAMICA MEDIDA'] <= row['BANDA DE ERROR DINAMICA RECOMENDADA'] else 2 if row['BANDA DE ERROR DINAMICA MEDIDA'] <= row['BANDA DE ERROR DINAMICA RECOMENDADA'] * 1.2 else 3
        p_l = 1 if row['LINEALIDAD DINAMICA MEDIDA'] <= row['LINEALIDAD DINAMICA RECOMENDADA'] else 2 if row['LINEALIDAD DINAMICA MEDIDA'] <= row['LINEALIDAD DINAMICA RECOMENDADA'] * 1.3 else 3
        
        total = p_f + p_c + p_b + p_l
        if total <= 4:
            return "ACEPTABLE"
        elif total <= 7:
            return "ACEPTABLE CON COMENTARIOS"
        else:
            return "ALERTA"
    
    y = df_clean.apply(calcular_estado, axis=1)
    print(f"   Clases calculadas: {y.value_counts().to_dict()}")

# 8. Codificar etiquetas
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\n📊 Clases codificadas:")
for i, clase in enumerate(label_encoder.classes_):
    print(f"   {i} -> {clase}")

# 9. Dividir datos (70/30 con estratificación)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\n📊 División de datos:")
print(f"   Entrenamiento: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Prueba: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

# 10. Escalar datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ Datos escalados (8 variables numéricas)")

# 11. Entrenar modelo
print("\n🌲 Entrenando Random Forest...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# 12. Evaluar modelo
y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊 Resultados del modelo:")
print(f"   Accuracy: {accuracy:.2%}")

print(f"\n📋 Reporte de clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 13. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Matriz de confusión:")
print("                 Predicho")
print("              " + "  ".join(label_encoder.classes_))
for i, clase in enumerate(label_encoder.classes_):
    print(f"Real: {clase:<20} {cm[i]}")

# 14. Guardar modelo (formato diccionario)
print("\n💾 Guardando modelo...")

model_data = {
    'model': rf_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'features': features,  # Guardar las 8 features usadas
    'is_trained': True,
    'model_type': 'random_forest'
}

joblib.dump(model_data, MODEL_PATH)
print(f"   ✅ Modelo guardado en: {MODEL_PATH}")

# Guardar componentes por separado (respaldo)
joblib.dump(scaler, SCALER_PATH)
joblib.dump(label_encoder, ENCODER_PATH)
print(f"   ✅ Scaler guardado en: {SCALER_PATH}")
print(f"   ✅ LabelEncoder guardado en: {ENCODER_PATH}")

print("\n" + "=" * 70)
print("✅ ENTRENAMIENTO DE RANDOM FOREST COMPLETADO")
print("=" * 70)