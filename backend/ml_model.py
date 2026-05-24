"""
MÓDULO DE MACHINE LEARNING - XGBOOST
Clase para diagnóstico de válvulas de control
"""

import pandas as pd
import numpy as np
import joblib
import os
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DiagnosticModel:
    """Modelo XGBoost para diagnóstico de válvulas de control"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.features = [
            'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
            'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
            'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
            'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
        ]
        self.is_trained = False
    
    def cargar(self, path):
        """Carga el modelo entrenado desde un archivo pickle"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el archivo: {path}")
        
        data = joblib.load(path)
        
        if isinstance(data, dict):
            self.model = data.get('model')
            self.scaler = data.get('scaler')
            self.label_encoder = data.get('label_encoder')
            self.features = data.get('features', self.features)
            self.is_trained = data.get('is_trained', False)
            print(f"✅ Modelo cargado desde diccionario: {path}")
            if self.label_encoder:
                print(f"   Clases: {self.label_encoder.classes_.tolist()}")
        else:
            self.model = data
            self.is_trained = True
            print(f"✅ Modelo cargado (formato directo): {path}")
        
        return self.is_trained
    
    def load(self, path):
        """Alias para cargar"""
        return self.cargar(path)
    
    def predecir(self, mediciones):
        """Predice el estado de una válvula individual"""
        if not self.is_trained or self.model is None:
            raise ValueError("Modelo no entrenado o no cargado")
        
        # Convertir mediciones a formato numérico
        X_dict = {}
        for feature in self.features:
            val = mediciones.get(feature, 0)
            if isinstance(val, str):
                val = float(val.replace(',', '.'))
            X_dict[feature] = float(val) if val is not None else 0
        
        # Crear DataFrame
        X = pd.DataFrame([X_dict])[self.features]
        
        # Escalar
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Predecir
        pred_encoded = self.model.predict(X_scaled)[0]
        
        # Decodificar
        if self.label_encoder:
            resultado = self.label_encoder.inverse_transform([pred_encoded])[0]
        else:
            resultado = str(pred_encoded)
        
        return resultado
    
    def calcular_recomendaciones(self, row):
        """Genera recomendaciones técnicas"""
        recomendaciones = {}
        
        try:
            # Fricción
            friccion_rec = float(str(row.get('FRICCION RECOMENDADA', 0)).replace(',', '.'))
            friccion_med = float(str(row.get('FRICCION MEDIDA', 0)).replace(',', '.'))
            desv_f = abs(friccion_med - friccion_rec)
            
            if desv_f > 10:
                recomendaciones['friccion'] = "⚠️ Inspección urgente de packing. Riesgo de fuga o atascamiento."
            elif desv_f > 5:
                recomendaciones['friccion'] = "📌 Revisar packing: posible falta de lubricación o desgaste."
            else:
                recomendaciones['friccion'] = "✅ Fricción dentro de especificaciones."
            
            # Carga
            carga_rec = float(str(row.get('TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 0)).replace(',', '.'))
            carga_med = float(str(row.get('TORQUE - CARGA EN EL ASIENTO MEDIDA', 0)).replace(',', '.'))
            
            if carga_rec > 0:
                porcentaje = (carga_med / carga_rec - 1) * 100
                if porcentaje < -15:
                    recomendaciones['carga'] = "⚠️ Inspección urgente. Probable fuga interna por desgaste severo."
                elif porcentaje < -10:
                    recomendaciones['carga'] = "📌 Verificar estado de internos (asiento y tapón). Posible desgaste."
                elif porcentaje > 25:
                    recomendaciones['carga'] = "⚠️ Riesgo de daños prematuros en asiento y tapón. Programar revisión."
                elif porcentaje > 15:
                    recomendaciones['carga'] = "📌 Monitorear tendencia. La sobrecarga puede generar desgaste prematuro."
                else:
                    recomendaciones['carga'] = "✅ Carga dentro de especificaciones."
            
            # Banda muerta
            banda_rec = float(str(row.get('BANDA DE ERROR DINAMICA RECOMENDADA', 0)).replace(',', '.'))
            banda_med = float(str(row.get('BANDA DE ERROR DINAMICA MEDIDA', 0)).replace(',', '.'))
            
            if banda_rec > 0:
                ratio = banda_med / banda_rec
                if ratio > 1.2:
                    recomendaciones['banda'] = "⚠️ Revisar urgentemente el posicionador. La válvula no responde adecuadamente."
                elif ratio > 1.0:
                    recomendaciones['banda'] = "📌 Verificar calibración del posicionador."
                else:
                    recomendaciones['banda'] = "✅ Banda muerta dentro de especificaciones."
            
            # Linealidad
            lin_rec = float(str(row.get('LINEALIDAD DINAMICA RECOMENDADA', 0)).replace(',', '.'))
            lin_med = float(str(row.get('LINEALIDAD DINAMICA MEDIDA', 0)).replace(',', '.'))
            
            if lin_rec > 0:
                ratio = lin_med / lin_rec
                if ratio > 1.3:
                    recomendaciones['linealidad'] = "⚠️ Revisar urgentemente el posicionador y su acople."
                elif ratio > 1.0:
                    recomendaciones['linealidad'] = "📌 Verificar caracterización del posicionador."
                else:
                    recomendaciones['linealidad'] = "✅ Linealidad dentro de especificaciones."
                
        except Exception as e:
            print(f"Error en recomendaciones: {e}")
        
        return recomendaciones
    
    def guardar(self, path):
        """Guarda el modelo en formato diccionario"""
        if not self.is_trained:
            raise ValueError("No hay modelo para guardar")
        
        data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'features': self.features,
            'is_trained': self.is_trained
        }
        joblib.dump(data, path)
        print(f"✅ Modelo guardado en: {path}")