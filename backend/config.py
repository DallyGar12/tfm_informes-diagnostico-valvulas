# ============================================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================================

import os

# Obtener la raíz del proyecto (dos niveles arriba de backend)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# RUTAS DE ARCHIVOS
# ============================================================================

# Archivo de datos de válvulas (INPUT)
DATA_PATH = os.path.join(BASE_DIR, 'Data', 'Dataset valvulas de control SK.csv')

# Archivo donde se guardará el modelo entrenado (OUTPUT)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'modelo_entrenado.pkl')

# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

API_HOST = '0.0.0.0'
API_PORT = 5000
API_DEBUG = True

# ============================================================================
# CONFIGURACIÓN DEL MODELO XGBOOST
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.3

# Parámetros de XGBoost
XGB_PARAMS = {
    'n_estimators': 100,      # Número de árboles
    'max_depth': 4,            # Profundidad máxima (evita sobreajuste)
    'learning_rate': 0.1,      # Tasa de aprendizaje
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'mlogloss'  # Métrica para clasificación multiclase
}