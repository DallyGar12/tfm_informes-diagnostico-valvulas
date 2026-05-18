# ============================================================================
# API PRINCIPAL - DIAGNÓSTICO DE VÁLVULAS CON XGBOOST
# ============================================================================

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_model import DiagnosticModel
import config

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

app = Flask(__name__)
CORS(app)

modelo = DiagnosticModel()
modelo_cargado = False

if os.path.exists(config.MODEL_PATH):
    try:
        modelo.cargar(config.MODEL_PATH)
        modelo_cargado = True
        print("✅ Modelo XGBoost cargado correctamente")
    except Exception as e:
        print(f"⚠️ Error cargando modelo: {e}")
else:
    print("⚠️ No se encontró modelo entrenado. Usa /api/admin/entrenar para entrenar")


# ============================================================================
# FUNCIÓN PARA CONVERTIR COMAS A PUNTOS
# ============================================================================

def convertir_df_a_numeros(df):
    """Convierte las columnas numéricas de un DataFrame, manejando comas decimales"""
    columnas_numericas = [
        'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
        'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
        'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
        'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
    ]
    
    df_resultado = df.copy()
    for col in columnas_numericas:
        if col in df_resultado.columns:
            df_resultado[col] = df_resultado[col].astype(str).str.replace(',', '.').str.strip()
            df_resultado[col] = pd.to_numeric(df_resultado[col], errors='coerce')
    return df_resultado


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'API funcionando con XGBoost'})


@app.route('/api/valvulas', methods=['GET'])
def obtener_valvulas():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos cargados'}), 404
    
    df = pd.read_csv(config.DATA_PATH, sep=';', encoding='utf-8')
    df = convertir_df_a_numeros(df)
    
    resultados = []
    for idx, row in df.iterrows():
        mediciones = {
            'FRICCION RECOMENDADA': row['FRICCION RECOMENDADA'],
            'FRICCION MEDIDA': row['FRICCION MEDIDA'],
            'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'],
            'TORQUE - CARGA EN EL ASIENTO MEDIDA': row['TORQUE - CARGA EN EL ASIENTO MEDIDA'],
            'BANDA DE ERROR DINAMICA RECOMENDADA': row['BANDA DE ERROR DINAMICA RECOMENDADA'],
            'BANDA DE ERROR DINAMICA MEDIDA': row['BANDA DE ERROR DINAMICA MEDIDA'],
            'LINEALIDAD DINAMICA RECOMENDADA': row['LINEALIDAD DINAMICA RECOMENDADA'],
            'LINEALIDAD DINAMICA MEDIDA': row['LINEALIDAD DINAMICA MEDIDA']
        }
        
        if modelo_cargado:
            try:
                estado = modelo.predecir(mediciones)
            except Exception as e:
                estado = f"ERROR: {str(e)[:50]}"
        else:
            estado = "MODELO NO ENTRENADO"
        
        resultados.append({
            'id': idx,
            'name': row.get('TAG CUERPO', f'VALVULA-{idx:03d}'),
            'status': estado,
            'measurements': {
                'friccion': {'measured': float(row['FRICCION MEDIDA']), 'recommended': float(row['FRICCION RECOMENDADA'])},
                'carga': {'measured': float(row['TORQUE - CARGA EN EL ASIENTO MEDIDA']), 'recommended': float(row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'])},
                'banda': {'measured': float(row['BANDA DE ERROR DINAMICA MEDIDA']), 'recommended': float(row['BANDA DE ERROR DINAMICA RECOMENDADA'])},
                'linealidad': {'measured': float(row['LINEALIDAD DINAMICA MEDIDA']), 'recommended': float(row['LINEALIDAD DINAMICA RECOMENDADA'])}
            }
        })
    
    return jsonify(resultados)


@app.route('/api/valvulas/<int:valvula_id>', methods=['GET'])
def obtener_valvula_detalle(valvula_id):
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos cargados'}), 404
    
    df = pd.read_csv(config.DATA_PATH, sep=';', encoding='utf-8')
    df = convertir_df_a_numeros(df)
    
    if valvula_id >= len(df):
        return jsonify({'error': 'Válvula no encontrada'}), 404
    
    row = df.iloc[valvula_id]
    
    mediciones = {
        'FRICCION RECOMENDADA': row['FRICCION RECOMENDADA'],
        'FRICCION MEDIDA': row['FRICCION MEDIDA'],
        'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'],
        'TORQUE - CARGA EN EL ASIENTO MEDIDA': row['TORQUE - CARGA EN EL ASIENTO MEDIDA'],
        'BANDA DE ERROR DINAMICA RECOMENDADA': row['BANDA DE ERROR DINAMICA RECOMENDADA'],
        'BANDA DE ERROR DINAMICA MEDIDA': row['BANDA DE ERROR DINAMICA MEDIDA'],
        'LINEALIDAD DINAMICA RECOMENDADA': row['LINEALIDAD DINAMICA RECOMENDADA'],
        'LINEALIDAD DINAMICA MEDIDA': row['LINEALIDAD DINAMICA MEDIDA']
    }
    
    if modelo_cargado:
        try:
            estado = modelo.predecir(mediciones)
            recomendaciones = modelo.calcular_recomendaciones(row, estado)
        except Exception as e:
            estado = f"ERROR: {str(e)[:50]}"
            recomendaciones = {}
        
        if estado == "ALERTA":
            severidad = "ALERTA"
        elif estado == "ACEPTABLE CON COMENTARIOS":
            severidad = "ACEPTABLE CON COMENTARIOS"
        else:
            severidad = "ACEPTABLE"
    else:
        estado = "Modelo no entrenado"
        severidad = "ACEPTABLE"
        recomendaciones = {}
    
    def get_sev(med, rec):
        try:
            desv = abs(float(med) - float(rec))
            if desv > 10:
                return "severo"
            elif desv > 5:
                return "moderado"
            return "normal"
        except:
            return "normal"
    
    return jsonify({
        'id': valvula_id,
        'name': row.get('TAG CUERPO', f'VALVULA-{valvula_id:03d}'),
        'overallSeverity': severidad,
        'status': estado,
        'measurements': [
            {
                'name': 'Fricción',
                'measured': float(row['FRICCION MEDIDA']) if pd.notna(row['FRICCION MEDIDA']) else 0,
                'recommended': float(row['FRICCION RECOMENDADA']) if pd.notna(row['FRICCION RECOMENDADA']) else 0,
                'unit': 'unidades',
                'severity': get_sev(row['FRICCION MEDIDA'], row['FRICCION RECOMENDADA']),
                'recommendation': recomendaciones.get('friccion', '')
            },
            {
                'name': 'Carga en Asiento/Torque',
                'measured': float(row['TORQUE - CARGA EN EL ASIENTO MEDIDA']) if pd.notna(row['TORQUE - CARGA EN EL ASIENTO MEDIDA']) else 0,
                'recommended': float(row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA']) if pd.notna(row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA']) else 0,
                'unit': 'unidades',
                'severity': 'normal',
                'recommendation': recomendaciones.get('carga', '')
            },
            {
                'name': 'Banda Muerta',
                'measured': float(row['BANDA DE ERROR DINAMICA MEDIDA']) if pd.notna(row['BANDA DE ERROR DINAMICA MEDIDA']) else 0,
                'recommended': float(row['BANDA DE ERROR DINAMICA RECOMENDADA']) if pd.notna(row['BANDA DE ERROR DINAMICA RECOMENDADA']) else 0,
                'unit': 'unidades',
                'severity': get_sev(row['BANDA DE ERROR DINAMICA MEDIDA'], row['BANDA DE ERROR DINAMICA RECOMENDADA']),
                'recommendation': recomendaciones.get('banda', '')
            },
            {
                'name': 'Linealidad Dinámica',
                'measured': float(row['LINEALIDAD DINAMICA MEDIDA']) if pd.notna(row['LINEALIDAD DINAMICA MEDIDA']) else 0,
                'recommended': float(row['LINEALIDAD DINAMICA RECOMENDADA']) if pd.notna(row['LINEALIDAD DINAMICA RECOMENDADA']) else 0,
                'unit': 'unidades',
                'severity': get_sev(row['LINEALIDAD DINAMICA MEDIDA'], row['LINEALIDAD DINAMICA RECOMENDADA']),
                'recommendation': recomendaciones.get('linealidad', '')
            }
        ],
        'recommendation': f"Válvula en estado {estado}",
        'actionPlan': recomendaciones
    })


@app.route('/api/resumen', methods=['GET'])
def obtener_resumen():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos cargados'}), 404
    
    df = pd.read_csv(config.DATA_PATH, sep=';', encoding='utf-8')
    df = convertir_df_a_numeros(df)
    
    if not modelo_cargado:
        return jsonify({
            'total': len(df),
            'alerta': 0,
            'aceptable_com': 0,
            'aceptable': len(df),
            'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'modelo_cargado': False
        })
    
    alerta = 0
    aceptable_com = 0
    aceptable = 0
    
    for idx, row in df.iterrows():
        mediciones = {
            'FRICCION RECOMENDADA': row['FRICCION RECOMENDADA'],
            'FRICCION MEDIDA': row['FRICCION MEDIDA'],
            'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'],
            'TORQUE - CARGA EN EL ASIENTO MEDIDA': row['TORQUE - CARGA EN EL ASIENTO MEDIDA'],
            'BANDA DE ERROR DINAMICA RECOMENDADA': row['BANDA DE ERROR DINAMICA RECOMENDADA'],
            'BANDA DE ERROR DINAMICA MEDIDA': row['BANDA DE ERROR DINAMICA MEDIDA'],
            'LINEALIDAD DINAMICA RECOMENDADA': row['LINEALIDAD DINAMICA RECOMENDADA'],
            'LINEALIDAD DINAMICA MEDIDA': row['LINEALIDAD DINAMICA MEDIDA']
        }
        
        try:
            estado = modelo.predecir(mediciones)
            if estado == "ALERTA":
                alerta += 1
            elif estado == "ACEPTABLE CON COMENTARIOS":
                aceptable_com += 1
            else:
                aceptable += 1
        except Exception as e:
            print(f"Error en fila {idx}: {e}")
            continue
    
    return jsonify({
        'total': len(df),
        'alerta': alerta,
        'aceptable_com': aceptable_com,
        'aceptable': aceptable,
        'fecha_actualizacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo_cargado': modelo_cargado
    })


# ============================================================================
# ENDPOINTS ADMIN
# ============================================================================

@app.route('/api/admin/entrenar', methods=['POST'])
def entrenar_modelo():
    global modelo_cargado, modelo
    
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No se encontró el archivo de datos'}), 404
    
    try:
        df = pd.read_csv(config.DATA_PATH, sep=';', encoding='utf-8')
        df = convertir_df_a_numeros(df)
        
        modelo = DiagnosticModel()
        accuracy = modelo.entrenar(df)
        
        modelo.guardar(config.MODEL_PATH)
        modelo_cargado = True
        
        return jsonify({
            'success': True,
            'message': f'Modelo XGBoost entrenado correctamente con {len(df)} válvulas',
            'accuracy': float(accuracy)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/admin/datos', methods=['POST'])
def cargar_datos():
    if 'file' not in request.files:
        return jsonify({'error': 'No se envió ningún archivo'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'El archivo debe ser CSV'}), 400
    
    os.makedirs(os.path.dirname(config.DATA_PATH), exist_ok=True)
    file.save(config.DATA_PATH)
    
    return jsonify({'success': True, 'message': f'Datos cargados: {file.filename}'})


@app.route('/api/admin/exportar', methods=['GET'])
def exportar_informe():
    if not os.path.exists(config.DATA_PATH):
        return jsonify({'error': 'No hay datos'}), 404
    
    df = pd.read_csv(config.DATA_PATH, sep=';', encoding='utf-8')
    df = convertir_df_a_numeros(df)
    
    diagnosticos = []
    for idx, row in df.iterrows():
        mediciones = {
            'FRICCION RECOMENDADA': row['FRICCION RECOMENDADA'],
            'FRICCION MEDIDA': row['FRICCION MEDIDA'],
            'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'],
            'TORQUE - CARGA EN EL ASIENTO MEDIDA': row['TORQUE - CARGA EN EL ASIENTO MEDIDA'],
            'BANDA DE ERROR DINAMICA RECOMENDADA': row['BANDA DE ERROR DINAMICA RECOMENDADA'],
            'BANDA DE ERROR DINAMICA MEDIDA': row['BANDA DE ERROR DINAMICA MEDIDA'],
            'LINEALIDAD DINAMICA RECOMENDADA': row['LINEALIDAD DINAMICA RECOMENDADA'],
            'LINEALIDAD DINAMICA MEDIDA': row['LINEALIDAD DINAMICA MEDIDA']
        }
        
        if modelo_cargado:
            try:
                estado = modelo.predecir(mediciones)
            except:
                estado = "ERROR"
        else:
            estado = "MODELO NO ENTRENADO"
        
        diagnosticos.append({
            'TAG': row.get('TAG CUERPO', f'VALVULA-{idx:03d}'),
            'Estado': estado,
            'Friccion_Medida': row['FRICCION MEDIDA'],
            'Friccion_Recomendada': row['FRICCION RECOMENDADA'],
            'Carga_Medida': row['TORQUE - CARGA EN EL ASIENTO MEDIDA'],
            'Carga_Recomendada': row['TORQUE - CARGA EN EL ASIENTO RECOMENDADA'],
            'Banda_Medida': row['BANDA DE ERROR DINAMICA MEDIDA'],
            'Banda_Recomendada': row['BANDA DE ERROR DINAMICA RECOMENDADA'],
            'Linealidad_Medida': row['LINEALIDAD DINAMICA MEDIDA'],
            'Linealidad_Recomendada': row['LINEALIDAD DINAMICA RECOMENDADA']
        })
    
    df_resultado = pd.DataFrame(diagnosticos)
    
    reports_dir = os.path.join(os.path.dirname(config.DATA_PATH), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    output_path = os.path.join(reports_dir, f"informe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df_resultado.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    return send_file(output_path, as_attachment=True, download_name=os.path.basename(output_path))


@app.route('/api/admin/status', methods=['GET'])
def obtener_status():
    datos_existen = os.path.exists(config.DATA_PATH)
    modelo_existe = os.path.exists(config.MODEL_PATH)
    
    return jsonify({
        'datos_cargados': datos_existen,
        'modelo_entrenado': modelo_existe and modelo_cargado,
        'version_api': '1.0.0',
        'modelo_tipo': 'XGBoost'
    })


# ============================================================================
# INICIO
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 INICIANDO API DE DIAGNÓSTICO DE VÁLVULAS CON XGBOOST")
    print("=" * 60)
    print(f"📁 DATA_PATH: {config.DATA_PATH}")
    print(f"📁 MODEL_PATH: {config.MODEL_PATH}")
    print(f"📁 CSV existe: {os.path.exists(config.DATA_PATH)}")
    print("=" * 60)
    
    app.run(debug=config.API_DEBUG, host=config.API_HOST, port=config.API_PORT)