"""
CONFIGURACIÓN DEL SISTEMA
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Rutas de archivos
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'modelo_xgb.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'models', 'scaler_xgb.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'encoder_xgb.pkl')

# Rutas para Random Forest (respaldo)
MODEL_RF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_rf.pkl')
SCALER_RF_PATH = os.path.join(BASE_DIR, 'models', 'scaler_rf.pkl')
ENCODER_RF_PATH = os.path.join(BASE_DIR, 'models', 'encoder_rf.pkl')

# Configuración de la API
API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# Configuración del modelo
RANDOM_STATE = 42
TEST_SIZE = 0.3