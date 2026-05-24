"""
CREAR DIAGNÓSTICO EXPERTO - Ground truth para 199 válvulas
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Cambia esta ruta a tu archivo de 199 válvulas
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'dataset_199_valvulas.csv')
OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')

print("=" * 70)
print("CREANDO GROUND TRUTH - DIAGNÓSTICO EXPERTO")
print("=" * 70)

print(f"\n📂 Cargando datos desde: {DATA_PATH}")
df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8')
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

def calcular_diagnostico(row):
    # Fricción
    diff_f = abs(row['FRICCION MEDIDA'] - row['FRICCION RECOMENDADA'])
    p_f = 1 if diff_f <= 5 else 2 if diff_f <= 10 else 3
    
    # Carga
    p_c = 1
    if row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] > 0:
        pct = abs((row['TORQUE - CARGA EN EL ASIENTO MEDIDA'] / row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] - 1) * 100)
        p_c = 1 if pct <= 10 else 2 if pct <= 25 else 3
    
    # Banda muerta
    p_b = 1
    if row['BANDA DE ERROR DINAMICA RECOMENDADA'] > 0:
        ratio = row['BANDA DE ERROR DINAMICA MEDIDA'] / row['BANDA DE ERROR DINAMICA RECOMENDADA']
        p_b = 1 if ratio <= 1.0 else 2 if ratio <= 1.2 else 3
    
    # Linealidad
    p_l = 1
    if row['LINEALIDAD DINAMICA RECOMENDADA'] > 0:
        ratio = row['LINEALIDAD DINAMICA MEDIDA'] / row['LINEALIDAD DINAMICA RECOMENDADA']
        p_l = 1 if ratio <= 1.0 else 2 if ratio <= 1.3 else 3
    
    total = p_f + p_c + p_b + p_l
    if total <= 4:
        return "ACEPTABLE"
    elif total <= 7:
        return "ACEPTABLE CON COMENTARIOS"
    else:
        return "ALERTA"

df['estado_real'] = df.apply(calcular_diagnostico, axis=1)

print("\n📊 Distribución del diagnóstico experto:")
print(df['estado_real'].value_counts())

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Ground truth guardado en: {OUTPUT_PATH}")