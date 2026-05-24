"""
SCRIPT DE PRUEBA - Verifica que los modelos funcionan correctamente
"""

import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

BASE_DIR = r"C:\Users\dally\OneDrive - UNIR\TRABAJO FIN DE MASTER\tfm_informes-diagnostico-valvulas"
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')
MODEL_RF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_rf.pkl')
MODEL_XGB_PATH = os.path.join(BASE_DIR, 'models', 'modelo_xgb.pkl')

print("=" * 60)
print("PRUEBA DE MODELOS")
print("=" * 60)

# Cargar datos
df = pd.read_csv(DATA_PATH)
y_true = df['estado_real']
print(f"\n✅ Datos cargados: {len(df)} filas")
print(f"   Clases reales: {y_true.value_counts().to_dict()}")

# Probar Random Forest
print("\n" + "-" * 40)
print("Probando Random Forest...")

if os.path.exists(MODEL_RF_PATH):
    try:
        data = joblib.load(MODEL_RF_PATH)
        if isinstance(data, dict):
            model = data['model']
            scaler = data.get('scaler')
            print("   ✅ Modelo cargado (formato diccionario)")
        else:
            model = data
            scaler = None
            print("   ✅ Modelo cargado (formato directo)")
        
        # Preparar features
        features = [col for col in df.columns if col not in ['estado_real', 'estado_automatico']]
        X = df[features]
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        
        y_pred = model.predict(X_scaled)
        
        # Verificar predicciones
        print(f"   📊 Predicciones: {pd.Series(y_pred).value_counts().to_dict()}")
        print(f"   ✅ Longitud: {len(y_pred)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("   ❌ No existe modelo_rf.pkl")

# Probar XGBoost
print("\n" + "-" * 40)
print("Probando XGBoost...")

if os.path.exists(MODEL_XGB_PATH):
    try:
        data = joblib.load(MODEL_XGB_PATH)
        if isinstance(data, dict):
            model = data['model']
            scaler = data.get('scaler')
            print("   ✅ Modelo cargado (formato diccionario)")
        else:
            model = data
            scaler = None
            print("   ✅ Modelo cargado (formato directo)")
        
        # Preparar features
        features = [col for col in df.columns if col not in ['estado_real', 'estado_automatico']]
        X = df[features]
        
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        
        y_pred = model.predict(X_scaled)
        
        # Verificar predicciones
        print(f"   📊 Predicciones: {pd.Series(y_pred).value_counts().to_dict()}")
        print(f"   ✅ Longitud: {len(y_pred)}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
else:
    print("   ❌ No existe modelo_xgb.pkl")

print("\n" + "=" * 60)