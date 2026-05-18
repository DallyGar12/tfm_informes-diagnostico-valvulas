# ============================================================================
# MÓDULO DE MACHINE LEARNING PARA DIAGNÓSTICO DE VÁLVULAS
# MODELO: XGBOOST (EXTREME GRADIENT BOOSTING)
# ============================================================================

import pandas as pd
import numpy as np
import joblib
import os
import re
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Configuración
RANDOM_STATE = 42
TEST_SIZE = 0.3

# Parámetros de XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 4,
    'learning_rate': 0.1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'
}


# ============================================================================
# FUNCIÓN PARA CONVERTIR TEXTOS A NÚMEROS (maneja comas decimales)
# ============================================================================

def convertir_a_numero(valor):
    """Convierte un valor a número, manejando comas decimales (ej: '2,42' -> 2.42)"""
    if pd.isna(valor):
        return np.nan
    if isinstance(valor, (int, float)):
        return float(valor)
    if isinstance(valor, str):
        valor_limpio = valor.strip()
        valor_limpio = valor_limpio.replace(',', '.')
        valor_limpio = valor_limpio.replace(' ', '')
        valor_limpio = valor_limpio.replace('%', '')
        valor_limpio = re.sub(r'[^0-9.-]', '', valor_limpio)
        if valor_limpio == '' or valor_limpio == '-':
            return np.nan
        try:
            num = float(valor_limpio)
            if num == int(num):
                return int(num)
            return num
        except:
            return np.nan
    return np.nan


def convertir_columnas_a_numeros(df, columnas):
    """Convierte múltiples columnas a números, manejando comas decimales"""
    df_resultado = df.copy()
    for col in columnas:
        if col in df_resultado.columns:
            df_resultado[col] = df_resultado[col].astype(str).str.replace(',', '.').str.strip()
            df_resultado[col] = pd.to_numeric(df_resultado[col], errors='coerce')
    return df_resultado


def imputar_valores_nulos(df, columnas):
    """
    Reemplaza los valores nulos con la mediana de cada columna.
    Retorna el DataFrame con los valores imputados y un diccionario con las medianas.
    """
    df_imputado = df.copy()
    medianas = {}
    
    for col in columnas:
        if col in df_imputado.columns:
            # Calcular la mediana (ignorando nulos)
            mediana = df_imputado[col].median()
            medianas[col] = mediana
            # Contar nulos antes
            nulos_antes = df_imputado[col].isnull().sum()
            # Reemplazar nulos con la mediana
            df_imputado[col] = df_imputado[col].fillna(mediana)
            # Contar nulos después
            nulos_despues = df_imputado[col].isnull().sum()
            if nulos_antes > 0:
                print(f"   📊 {col}: {nulos_antes} valores nulos → reemplazados con mediana = {mediana:.2f}")
    
    return df_imputado, medianas


# ============================================================================
# CLASE DEL MODELO CON XGBOOST
# ============================================================================

class DiagnosticModel:
    """
    Modelo de clasificación XGBoost para diagnóstico de válvulas de control.
    Predice el estado de la válvula: ACEPTABLE, ACEPTABLE CON COMENTARIOS, ALERTA
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.medianas = None  # Guardar medianas para usar en predicciones
        self.features = [
            'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
            'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
            'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
            'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
        ]
        self.is_trained = False
    
    # --------------------------------------------------------------------------
    # MÉTODOS PRIVADOS
    # --------------------------------------------------------------------------
    
    def _calcular_estado_real(self, df):
        """
        Calcula el estado real de cada válvula según criterios técnicos ampliados.
        Retorna: ACEPTABLE, ACEPTABLE CON COMENTARIOS, ALERTA
        """
        df_temp = df.copy()
        
        # Calcular niveles individuales (1=Normal, 2=Moderado, 3=Severo)
        df_temp['nivel_friccion'] = df_temp.apply(
            lambda x: 1 if abs(x['FRICCION MEDIDA'] - x['FRICCION RECOMENDADA']) <= 5 else
                      2 if abs(x['FRICCION MEDIDA'] - x['FRICCION RECOMENDADA']) <= 10 else 3,
            axis=1
        )
        
        df_temp['nivel_carga'] = df_temp.apply(
            lambda x: 1 if abs((x['TORQUE - CARGA EN EL ASIENTO MEDIDA'] / x['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] - 1) * 100) <= 10 else
                      2 if abs((x['TORQUE - CARGA EN EL ASIENTO MEDIDA'] / x['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] - 1) * 100) <= 25 else 3,
            axis=1
        )
        
        df_temp['nivel_banda'] = df_temp.apply(
            lambda x: 1 if x['BANDA DE ERROR DINAMICA MEDIDA'] <= x['BANDA DE ERROR DINAMICA RECOMENDADA'] else
                      2 if x['BANDA DE ERROR DINAMICA MEDIDA'] <= x['BANDA DE ERROR DINAMICA RECOMENDADA'] * 1.2 else 3,
            axis=1
        )
        
        df_temp['nivel_linealidad'] = df_temp.apply(
            lambda x: 1 if x['LINEALIDAD DINAMICA MEDIDA'] <= x['LINEALIDAD DINAMICA RECOMENDADA'] else
                      2 if x['LINEALIDAD DINAMICA MEDIDA'] <= x['LINEALIDAD DINAMICA RECOMENDADA'] * 1.3 else 3,
            axis=1
        )
        
        # Función para clasificar según combinación de niveles
        def get_estado(f, c, b, l):
            # ACEPTABLE (VERDE)
            if f == 1 and c == 1 and b == 1 and l == 1:
                return "ACEPTABLE"
            if f == 1 and c == 1 and b == 1 and l == 2:
                return "ACEPTABLE"
            if f == 1 and c == 1 and b == 2 and l == 1:
                return "ACEPTABLE"
            if f == 1 and c == 2 and b == 1 and l == 1:
                return "ACEPTABLE"
            if f == 2 and c == 1 and b == 1 and l == 1:
                return "ACEPTABLE"
            
            # ACEPTABLE CON COMENTARIOS (AMARILLO)
            if f == 1 and c == 1 and b == 2 and l == 2:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 2 and c == 2 and b == 1 and l == 1:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 1 and c == 2 and b == 2 and l == 1:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 2 and c == 1 and b == 1 and l == 2:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 2 and c == 2 and b == 2 and l == 2:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 1 and c == 2 and b == 1 and l == 2:
                return "ACEPTABLE CON COMENTARIOS"
            if f == 2 and c == 1 and b == 2 and l == 1:
                return "ACEPTABLE CON COMENTARIOS"
            
            # ALERTA (NARANJA)
            if f == 3 and c == 3 and b == 1 and l == 1:
                return "ALERTA"
            if f == 3 and c == 3 and b == 2 and l == 2:
                return "ALERTA"
            if f == 1 and c == 1 and b == 3 and l == 3:
                return "ALERTA"
            if f == 2 and c == 2 and b == 3 and l == 3:
                return "ALERTA"
            if f == 3 and c == 3 and b == 3 and l == 3:
                return "ALERTA"
            if f == 1 and c == 1 and b == 1 and l == 3:
                return "ALERTA"
            if f == 3 and c == 1 and b == 1 and l == 1:
                return "ALERTA"
            if f == 1 and c == 3 and b == 1 and l == 1:
                return "ALERTA"
            if f == 1 and c == 1 and b == 3 and l == 1:
                return "ALERTA"
            if f == 1 and c == 1 and b == 2 and l == 3:
                return "ALERTA"
            if f == 1 and c == 2 and b == 1 and l == 3:
                return "ALERTA"
            if f == 2 and c == 1 and b == 1 and l == 3:
                return "ALERTA"
            if f == 3 and c == 3 and b == 1 and l == 2:
                return "ALERTA"
            if f == 3 and c == 2 and b == 3 and l == 1:
                return "ALERTA"
            
            # Por defecto
            return "ACEPTABLE CON COMENTARIOS"
        
        return df_temp.apply(
            lambda x: get_estado(
                x['nivel_friccion'], 
                x['nivel_carga'], 
                x['nivel_banda'], 
                x['nivel_linealidad']
            ), axis=1
        )
    
    # --------------------------------------------------------------------------
    # MÉTODOS PÚBLICOS
    # --------------------------------------------------------------------------
    
    def entrenar(self, df):
        """
        Entrena el modelo XGBoost con los datos proporcionados.
        
        Args:
            df: DataFrame con las columnas de diagnóstico
        
        Returns:
            accuracy: Precisión del modelo en entrenamiento
        """
        print("\n" + "=" * 60)
        print("🔧 INICIANDO ENTRENAMIENTO DEL MODELO XGBOOST")
        print("=" * 60)
        
        # Validar columnas
        for col in self.features:
            if col not in df.columns:
                raise ValueError(f"La columna '{col}' no existe en el dataset")
        
        print(f"📊 Dataset original: {len(df)} filas")
        
        # PASO 1: Convertir a números (maneja comas decimales)
        print("\n📝 Convertiendo datos a formato numérico...")
        df_clean = convertir_columnas_a_numeros(df, self.features)
        
        # PASO 2: Imputar valores nulos con la mediana
        print("\n📝 Imputando valores nulos...")
        df_clean, self.medianas = imputar_valores_nulos(df_clean, self.features)
        
        # Verificar si quedaron nulos después de la imputación
        nulos_restantes = df_clean[self.features].isnull().sum().sum()
        if nulos_restantes > 0:
            print(f"⚠️ Aún hay {nulos_restantes} valores nulos. Eliminando filas...")
            df_clean = df_clean.dropna(subset=self.features)
        
        print(f"✅ Filas después de limpieza e imputación: {len(df_clean)}")
        
        # Mostrar estadísticas básicas
        print("\n📊 Estadísticas de los datos después de limpieza:")
        for col in self.features:
            print(f"   {col}: min={df_clean[col].min():.2f}, max={df_clean[col].max():.2f}, media={df_clean[col].mean():.2f}")
        
        # Preparar características
        X = df_clean[self.features]
        
        # Calcular variable objetivo
        print("\n📝 Calculando estados reales según criterios técnicos...")
        y = self._calcular_estado_real(df_clean.copy())
        
        # Mostrar distribución de estados
        print("\n📊 Distribución de estados en los datos:")
        print(y.value_counts())
        
        # Codificar etiquetas
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Escalar características
        print("\n📝 Escalando características...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Crear y entrenar modelo XGBoost
        print("\n📝 Entrenando modelo XGBoost...")
        self.model = xgb.XGBClassifier(**XGB_PARAMS)
        self.model.fit(X_scaled, y_encoded)
        
        self.is_trained = True
        
        # Calcular precisión en entrenamiento
        y_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y_encoded, y_pred)
        
        print("\n" + "=" * 60)
        print(f"✅ ENTRENAMIENTO COMPLETADO")
        print(f"   Precisión en entrenamiento: {accuracy:.2%}")
        print("=" * 60 + "\n")
        
        return accuracy
    
    def predecir(self, mediciones):
        """
        Predice el estado de una válvula individual.
        
        Args:
            mediciones: Diccionario con las 8 mediciones usando los nombres reales
        
        Returns:
            estado: "ACEPTABLE", "ACEPTABLE CON COMENTARIOS", o "ALERTA"
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        # Convertir valores a float (manejando comas)
        mediciones_convertidas = {}
        for key, value in mediciones.items():
            if isinstance(value, str):
                value = value.replace(',', '.')
                try:
                    mediciones_convertidas[key] = float(value)
                except:
                    mediciones_convertidas[key] = value
            else:
                mediciones_convertidas[key] = float(value) if value is not None else np.nan
        
        # Verificar y reemplazar valores nulos con la mediana si es necesario
        for key in self.features:
            if pd.isna(mediciones_convertidas.get(key)):
                if self.medianas and key in self.medianas:
                    mediciones_convertidas[key] = self.medianas[key]
                    print(f"⚠️ Valor nulo en {key} reemplazado con mediana: {self.medianas[key]:.2f}")
        
        X = pd.DataFrame([mediciones_convertidas])[self.features]
        X_scaled = self.scaler.transform(X)
        pred_encoded = self.model.predict(X_scaled)[0]
        return self.label_encoder.inverse_transform([pred_encoded])[0]
    
    def predecir_batch(self, df):
        """
        Predice el estado de múltiples válvulas.
        """
        if not self.is_trained:
            raise ValueError("El modelo no ha sido entrenado aún")
        
        df_clean = convertir_columnas_a_numeros(df, self.features)
        
        # Imputar valores nulos si es necesario
        for col in self.features:
            if self.medianas and col in self.medianas:
                df_clean[col] = df_clean[col].fillna(self.medianas[col])
        
        X = df_clean[self.features]
        X_scaled = self.scaler.transform(X)
        predicciones = self.model.predict(X_scaled)
        return self.label_encoder.inverse_transform(predicciones)
    
    def calcular_recomendaciones(self, row, estado=None):
        """
        Genera recomendaciones técnicas basadas en las mediciones.
        """
        recomendaciones = {}
        
        try:
            friccion_rec = float(str(row['FRICCION RECOMENDADA']).replace(',', '.'))
            friccion_med = float(str(row['FRICCION MEDIDA']).replace(',', '.'))
            carga_rec = float(str(row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA']).replace(',', '.'))
            carga_med = float(str(row['TORQUE - CARGA EN EL ASIENTO MEDIDA']).replace(',', '.'))
            banda_rec = float(str(row['BANDA DE ERROR DINAMICA RECOMENDADA']).replace(',', '.'))
            banda_med = float(str(row['BANDA DE ERROR DINAMICA MEDIDA']).replace(',', '.'))
            linealidad_rec = float(str(row['LINEALIDAD DINAMICA RECOMENDADA']).replace(',', '.'))
            linealidad_med = float(str(row['LINEALIDAD DINAMICA MEDIDA']).replace(',', '.'))
        except:
            return recomendaciones
        
        # Fricción
        desv_friccion = abs(friccion_med - friccion_rec)
        if desv_friccion > 10:
            recomendaciones['friccion'] = "Inspección urgente de packing. Riesgo de fuga o atascamiento."
        elif desv_friccion > 5:
            recomendaciones['friccion'] = "Revisar packing: posible falta de lubricación o desgaste."
        else:
            recomendaciones['friccion'] = "Fricción dentro de especificaciones."
        
        # Carga
        if carga_rec > 0:
            porcentaje_carga = (carga_med / carga_rec - 1) * 100
        else:
            porcentaje_carga = 0
            
        if porcentaje_carga < -15:
            recomendaciones['carga'] = "Inspección urgente. Probable fuga interna por desgaste severo."
        elif porcentaje_carga < -10:
            recomendaciones['carga'] = "Verificar estado de internos (asiento y tapón). Posible desgaste."
        elif porcentaje_carga > 25:
            recomendaciones['carga'] = "Riesgo de daños prematuros en asiento y tapón. Programar revisión."
        elif porcentaje_carga > 15:
            recomendaciones['carga'] = "Monitorear tendencia. La sobrecarga puede generar desgaste prematuro."
        else:
            recomendaciones['carga'] = "Carga dentro de especificaciones."
        
        # Banda muerta
        if banda_med > banda_rec * 1.2:
            recomendaciones['banda'] = "Revisar urgentemente el posicionador. La válvula no responde adecuadamente."
        elif banda_med > banda_rec:
            recomendaciones['banda'] = "Verificar calibración del posicionador."
        else:
            recomendaciones['banda'] = "Banda muerta dentro de especificaciones."
        
        # Linealidad
        if linealidad_med > linealidad_rec * 1.3:
            recomendaciones['linealidad'] = "Revisar urgentemente el posicionador y su acople."
        elif linealidad_med > linealidad_rec:
            recomendaciones['linealidad'] = "Verificar caracterización del posicionador."
        else:
            recomendaciones['linealidad'] = "Linealidad dentro de especificaciones."
        
        return recomendaciones
    
    def guardar(self, path):
        """
        Guarda el modelo entrenado en un archivo pickle.
        """
        if not self.is_trained:
            raise ValueError("No hay modelo entrenado para guardar")
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.features,
            'medianas': self.medianas,
            'is_trained': self.is_trained,
            'model_type': 'xgboost'
        }
        joblib.dump(data, path)
        print(f"✅ Modelo XGBoost guardado en: {path}")
    
    def cargar(self, path):
        """
        Carga un modelo entrenado desde un archivo pickle.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        self.features = data['features']
        self.medianas = data.get('medianas', None)  # Compatibilidad con versiones anteriores
        self.is_trained = data['is_trained']
        print(f"✅ Modelo XGBoost cargado desde: {path}")
        if 'model_type' in data:
            print(f"   Tipo de modelo: {data['model_type']}")


# ============================================================================
# PRUEBA RÁPIDA (solo si se ejecuta directamente)
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PRUEBA DEL MÓDULO ML CON XGBOOST")
    print("=" * 60)
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'FRICCION RECOMENDADA': np.random.uniform(50, 150, n),
        'FRICCION MEDIDA': np.random.uniform(40, 160, n),
        'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': np.random.uniform(100, 500, n),
        'TORQUE - CARGA EN EL ASIENTO MEDIDA': np.random.uniform(80, 600, n),
        'BANDA DE ERROR DINAMICA RECOMENDADA': np.random.uniform(2, 8, n),
        'BANDA DE ERROR DINAMICA MEDIDA': np.random.uniform(1, 12, n),
        'LINEALIDAD DINAMICA RECOMENDADA': np.random.uniform(1, 5, n),
        'LINEALIDAD DINAMICA MEDIDA': np.random.uniform(0.5, 7, n)
    })
    
    modelo = DiagnosticModel()
    accuracy = modelo.entrenar(df)
    print(f"🎯 Accuracy XGBoost: {accuracy:.2%}")