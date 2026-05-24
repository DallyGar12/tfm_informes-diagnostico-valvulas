"""
PREPROCESAMIENTO DE DATOS DE VÁLVULAS
Limpia los datos y verifica la columna de diagnóstico experto
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Dataset valvulas de control TY - Comparativa.csv')
PROCESSED_DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')

os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)

print("=" * 70)
print("PREPROCESAMIENTO DE DATOS")
print("=" * 70)

# 1. Cargar datos
print(f"\n📂 Cargando datos desde: {RAW_DATA_PATH}")
df = pd.read_csv(RAW_DATA_PATH, sep=';', encoding='utf-8')
print(f"✅ Datos originales: {df.shape[0]} filas, {df.shape[1]} columnas")

# 2. Verificar columnas necesarias
features = [
    'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
    'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
    'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
    'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
]

print("\n📋 Verificando columnas de features...")
for col in features:
    if col not in df.columns:
        print(f"   ❌ Columna faltante: {col}")
    else:
        print(f"   ✅ {col}")

# 3. Verificar columna de diagnóstico experto
posibles_expertos = ['DIAGNOSTICO_EXPERTO', 'DIAGNOSTICO', 'ESTADO', 'EXPERTO', 'CLASIFICACION']
columna_experto = None

for col in posibles_expertos:
    if col in df.columns:
        columna_experto = col
        break

if columna_experto:
    print(f"\n✅ Columna de diagnóstico experto encontrada: '{columna_experto}'")
    print(f"   Valores únicos: {df[columna_experto].unique()}")
    print(f"   Distribución:")
    print(df[columna_experto].value_counts())
else:
    print(f"\n⚠️ No se encontró columna de diagnóstico experto")
    print(f"   Se usará cálculo automático (solo para referencia)")

# 4. Convertir a números
print("\n📝 Convirtiendo datos a formato numérico...")
for col in features:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 5. Eliminar filas con nulos en features
df_clean = df.dropna(subset=features)
print(f"✅ Filas después de limpieza: {len(df_clean)}")

# 6. Calcular estado automático (para comparación, NO para entrenar)
def calcular_estado_automatico(row):
    """Calcula estado según fórmula - SOLO para referencia, NO para entrenar"""
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

if columna_experto:
    df_clean['estado_automatico'] = df_clean.apply(calcular_estado_automatico, axis=1)
    df_clean['estado_real'] = df_clean[columna_experto]
    
    # Comparar distribuciones
    print("\n📊 Comparación de distribuciones:")
    print("\nEstado real (experto):")
    print(df_clean['estado_real'].value_counts())
    print("\nEstado automático (fórmula):")
    print(df_clean['estado_automatico'].value_counts())
    
    # Verificar coincidencia
    coincidencia = (df_clean['estado_real'] == df_clean['estado_automatico']).mean()
    print(f"\n🔍 Coincidencia entre experto y fórmula: {coincidencia:.1%}")
    
    if coincidencia > 0.95:
        print("\n⚠️ ¡ALERTA! El diagnóstico experto es casi idéntico a la fórmula.")
        print("   El modelo aprenderá la fórmula, no patrones reales.")
    else:
        print("\n✅ Buena diferencia entre experto y fórmula. El modelo aprenderá patrones reales.")
else:
    df_clean['estado_real'] = df_clean.apply(calcular_estado_automatico, axis=1)
    print("\n⚠️ Usando estado automático como referencia (no ideal para ML real)")

# 7. Guardar datos procesados
df_clean.to_csv(PROCESSED_DATA_PATH, index=False, encoding='utf-8-sig')
print(f"\n✅ Datos procesados guardados en: {PROCESSED_DATA_PATH}")

# 8. Mostrar resumen
print("\n" + "=" * 70)
print("RESUMEN DE DATOS PROCESADOS")
print("=" * 70)
print(f"Total de registros: {len(df_clean)}")
print(f"Features numéricas: {len(features)}")
print(f"Variable objetivo: estado_real")
print(f"Clases: {df_clean['estado_real'].unique().tolist()}")
print("\n" + "=" * 70)