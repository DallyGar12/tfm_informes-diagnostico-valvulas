"""
API PRINCIPAL - DIAGNÓSTICO DE VÁLVULAS CON XGBOOST
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
import tempfile
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_model import DiagnosticModel
import config

app = Flask(__name__)
CORS(app)

# Inicializar modelo
modelo = DiagnosticModel()
modelo_cargado = False

def cargar_modelo():
    global modelo_cargado
    if os.path.exists(config.MODEL_PATH):
        try:
            modelo.cargar(config.MODEL_PATH)
            modelo_cargado = True
            print("✅ Modelo XGBoost cargado correctamente")
        except Exception as e:
            print(f"⚠️ Error cargando modelo: {e}")
    else:
        print("⚠️ No se encontró modelo entrenado")

cargar_modelo()

def convertir_df(df):
    columnas = modelo.features
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# ============================================================================
# ENDPOINTS PRINCIPALES
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'API funcionando'})

@app.route('/api/valvulas', methods=['GET'])
def get_valvulas():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos'}), 404
    
    df = pd.read_csv(config.DATA_PATH, encoding='utf-8-sig')
    df = convertir_df(df)
    
    resultados = []
    for idx, row in df.iterrows():
        mediciones = {col: row[col] if col in df.columns else 0 for col in modelo.features}
        
        if modelo_cargado:
            try:
                estado = modelo.predecir(mediciones)
            except:
                estado = "ERROR"
        else:
            estado = "MODELO NO ENTRENADO"
        
        nombre = row.get('TAG CUERPO', f'VALVULA-{idx:03d}')
        
        resultados.append({
            'id': idx,
            'name': str(nombre),
            'status': estado,
            'measurements': [
                {'name': 'Fricción', 'measured': float(row.get('FRICCION MEDIDA', 0)), 
                 'recommended': float(row.get('FRICCION RECOMENDADA', 0)), 'unit': 'unid'},
                {'name': 'Carga en Asiento/Torque', 'measured': float(row.get('TORQUE - CARGA EN EL ASIENTO MEDIDA', 0)),
                 'recommended': float(row.get('TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 0)), 'unit': 'unid'},
                {'name': 'Banda Muerta', 'measured': float(row.get('BANDA DE ERROR DINAMICA MEDIDA', 0)),
                 'recommended': float(row.get('BANDA DE ERROR DINAMICA RECOMENDADA', 0)), 'unit': '%'},
                {'name': 'Linealidad Dinámica', 'measured': float(row.get('LINEALIDAD DINAMICA MEDIDA', 0)),
                 'recommended': float(row.get('LINEALIDAD DINAMICA RECOMENDADA', 0)), 'unit': '%'}
            ]
        })
    
    return jsonify(resultados)

@app.route('/api/valvulas/<int:vid>', methods=['GET'])
def get_valvula(vid):
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos'}), 404
    
    df = pd.read_csv(config.DATA_PATH, encoding='utf-8-sig')
    df = convertir_df(df)
    
    if vid >= len(df):
        return jsonify({'error': 'Válvula no encontrada'}), 404
    
    row = df.iloc[vid]
    mediciones = {col: row[col] if col in df.columns else 0 for col in modelo.features}
    
    if modelo_cargado:
        try:
            estado = modelo.predecir(mediciones)
            recomendaciones = modelo.calcular_recomendaciones(row)
        except:
            estado = "ERROR"
            recomendaciones = {}
    else:
        estado = "MODELO NO ENTRENADO"
        recomendaciones = {}
    
    nombre = row.get('TAG CUERPO', f'VALVULA-{vid:03d}')
    
    def get_sev(med, rec):
        try:
            diff = abs(float(med) - float(rec))
            return "severo" if diff > 10 else "moderado" if diff > 5 else "normal"
        except:
            return "normal"
    
    return jsonify({
        'id': vid,
        'name': str(nombre),
        'status': estado,
        'measurements': [
            {'name': 'Fricción', 'measured': float(row.get('FRICCION MEDIDA', 0)),
             'recommended': float(row.get('FRICCION RECOMENDADA', 0)), 'unit': 'unid',
             'severity': get_sev(row.get('FRICCION MEDIDA', 0), row.get('FRICCION RECOMENDADA', 0))},
            {'name': 'Carga en Asiento/Torque', 'measured': float(row.get('TORQUE - CARGA EN EL ASIENTO MEDIDA', 0)),
             'recommended': float(row.get('TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 0)), 'unit': 'unid',
             'severity': 'normal'},
            {'name': 'Banda Muerta', 'measured': float(row.get('BANDA DE ERROR DINAMICA MEDIDA', 0)),
             'recommended': float(row.get('BANDA DE ERROR DINAMICA RECOMENDADA', 0)), 'unit': '%',
             'severity': get_sev(row.get('BANDA DE ERROR DINAMICA MEDIDA', 0), row.get('BANDA DE ERROR DINAMICA RECOMENDADA', 0))},
            {'name': 'Linealidad Dinámica', 'measured': float(row.get('LINEALIDAD DINAMICA MEDIDA', 0)),
             'recommended': float(row.get('LINEALIDAD DINAMICA RECOMENDADA', 0)), 'unit': '%',
             'severity': get_sev(row.get('LINEALIDAD DINAMICA MEDIDA', 0), row.get('LINEALIDAD DINAMICA RECOMENDADA', 0))}
        ],
        'recommendation': f"Válvula en estado {estado}",
        'actionPlan': recomendaciones
    })

@app.route('/api/resumen', methods=['GET'])
def get_resumen():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos'}), 404
    
    df = pd.read_csv(config.DATA_PATH, encoding='utf-8-sig')
    df = convertir_df(df)
    
    if not modelo_cargado:
        return jsonify({'total': len(df), 'alerta': 0, 'aceptable_com': 0, 'aceptable': len(df), 'modelo_cargado': False})
    
    alerta = aceptable_com = aceptable = 0
    for _, row in df.iterrows():
        mediciones = {col: row[col] if col in df.columns else 0 for col in modelo.features}
        try:
            estado = modelo.predecir(mediciones)
            if estado == "ALERTA":
                alerta += 1
            elif estado == "ACEPTABLE CON COMENTARIOS":
                aceptable_com += 1
            else:
                aceptable += 1
        except:
            continue
    
    return jsonify({'total': len(df), 'alerta': alerta, 'aceptable_com': aceptable_com, 'aceptable': aceptable, 'modelo_cargado': True})

# ============================================================================
# ENDPOINTS ADMIN
# ============================================================================

@app.route('/api/admin/status', methods=['GET'])
def admin_status():
    return jsonify({
        'datos_cargados': os.path.exists(config.DATA_PATH),
        'modelo_entrenado': modelo_cargado,
        'version_api': '2.0.0'
    })

@app.route('/api/admin/cargar_dataset', methods=['POST'])
def cargar_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo vacío'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Formato debe ser CSV'}), 400
    
    try:
        temp_path = os.path.join(tempfile.gettempdir(), 'temp_dataset.csv')
        file.save(temp_path)
        
        df = pd.read_csv(temp_path, encoding='utf-8-sig')
        
        columnas_necesarias = modelo.features
        faltantes = [c for c in columnas_necesarias if c not in df.columns]
        if faltantes:
            return jsonify({'error': f'Columnas faltantes: {faltantes}'}), 400
        
        df = convertir_df(df)
        
        os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
        df.to_csv(config.DATA_PATH, index=False, encoding='utf-8-sig')
        os.remove(temp_path)
        
        return jsonify({'success': True, 'message': f'Dataset cargado: {len(df)} filas', 'num_filas': len(df)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/entrenar', methods=['POST'])
def entrenar_modelo():
    global modelo_cargado, modelo
    
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos cargados'}), 404
    
    try:
        import xgboost as xgb
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(config.DATA_PATH, encoding='utf-8-sig')
        df = convertir_df(df)
        
        # Generar estado_real si no existe
        if 'estado_real' not in df.columns:
            def generar_estado(row):
                p_f = 1 if abs(row['FRICCION MEDIDA'] - row['FRICCION RECOMENDADA']) <= 5 else 2 if abs(row['FRICCION MEDIDA'] - row['FRICCION RECOMENDADA']) <= 10 else 3
                p_c = 1
                if row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] > 0:
                    pct = abs((row['TORQUE - CARGA EN EL ASIENTO MEDIDA'] / row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'] - 1) * 100)
                    p_c = 1 if pct <= 10 else 2 if pct <= 25 else 3
                p_b = 1
                if row['BANDA DE ERROR DINAMICA RECOMENDADA'] > 0:
                    ratio = row['BANDA DE ERROR DINAMICA MEDIDA'] / row['BANDA DE ERROR DINAMICA RECOMENDADA']
                    p_b = 1 if ratio <= 1.0 else 2 if ratio <= 1.2 else 3
                p_l = 1
                if row['LINEALIDAD DINAMICA RECOMENDADA'] > 0:
                    ratio = row['LINEALIDAD DINAMICA MEDIDA'] / row['LINEALIDAD DINAMICA RECOMENDADA']
                    p_l = 1 if ratio <= 1.0 else 2 if ratio <= 1.3 else 3
                total = p_f + p_c + p_b + p_l
                return "ACEPTABLE" if total <= 4 else "ACEPTABLE CON COMENTARIOS" if total <= 7 else "ALERTA"
            
            df['estado_real'] = df.apply(generar_estado, axis=1)
            df.to_csv(config.DATA_PATH, index=False, encoding='utf-8-sig')
        
        X = df[modelo.features]
        y = df['estado_real']
        
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
        xgb_model.fit(X_scaled, y_enc)
        
        modelo.model = xgb_model
        modelo.scaler = scaler
        modelo.label_encoder = le
        modelo.is_trained = True
        modelo_cargado = True
        
        modelo.guardar(config.MODEL_PATH)
        
        y_pred = xgb_model.predict(X_scaled)
        accuracy = (y_pred == y_enc).mean()
        
        return jsonify({'success': True, 'message': 'Modelo entrenado', 'accuracy': float(accuracy)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/exportar', methods=['GET'])
def exportar():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos'}), 404
    
    temp_path = os.path.join(tempfile.gettempdir(), f'informe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    
    df = pd.read_csv(config.DATA_PATH, encoding='utf-8-sig')
    df = convertir_df(df)
    
    if modelo_cargado:
        predicciones = []
        for _, row in df.iterrows():
            mediciones = {col: row[col] if col in df.columns else 0 for col in modelo.features}
            try:
                pred = modelo.predecir(mediciones)
            except:
                pred = "ERROR"
            predicciones.append(pred)
        df['PREDICCION_MODELO'] = predicciones
    
    df.to_csv(temp_path, index=False, encoding='utf-8-sig')
    return send_file(temp_path, as_attachment=True, download_name=os.path.basename(temp_path))

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 INICIANDO API DE DIAGNÓSTICO DE VÁLVULAS")
    print("=" * 60)
    print(f"📁 DATA: {config.DATA_PATH}")
    print(f"📁 MODELO: {config.MODEL_PATH}")
    print("=" * 60)
    app.run(debug=config.API_DEBUG, host=config.API_HOST, port=config.API_PORT)