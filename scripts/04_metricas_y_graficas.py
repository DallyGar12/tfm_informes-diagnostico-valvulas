"""
MÉTRICAS Y GRÁFICAS COMPARATIVAS - CON GRÁFICAS DE PASTEL
Genera todas las gráficas para el TFM:
1. Distribución de estados (barras) - Diagnóstico Experto
2. Distribución de estados (pastel) - Diagnóstico Experto
3. Comparación de pasteles (3 en 1) - Experto vs RF vs XGBoost
4. Comparación de métricas (accuracy, precision, recall, f1)
5. Matrices de confusión de ambos modelos
6. Importancia de features para Random Forest y XGBoost
7. Comparación de predicciones vs real
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'dataset_limpio.csv')
MODEL_RF_PATH = os.path.join(BASE_DIR, 'models', 'modelo_rf.pkl')
MODEL_XGB_PATH = os.path.join(BASE_DIR, 'models', 'modelo_xgb.pkl')

OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuración de gráficos
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colores para los estados
COLORES_ESTADOS = {
    'ACEPTABLE': '#2ECC71',           # Verde
    'ACEPTABLE CON COMENTARIOS': '#F1C40F',  # Amarillo
    'ALERTA': '#E74C3C'               # Rojo
}

# Colores para los modelos
COLORES_MODELOS = {
    'Random Forest': '#3498DB',  # Azul
    'XGBoost': '#E67E22'         # Naranja
}

# Nombres legibles para las features
FEATURE_NAMES = {
    'FRICCION RECOMENDADA': 'Fricción Recomendada',
    'FRICCION MEDIDA': 'Fricción Medida',
    'TORQUE - CARGA EN EL ASIENTO RECOMENDADA': 'Torque Recomendado',
    'TORQUE - CARGA EN EL ASIENTO MEDIDA': 'Torque Medido',
    'BANDA DE ERROR DINAMICA RECOMENDADA': 'Banda Muerta Recomendada',
    'BANDA DE ERROR DINAMICA MEDIDA': 'Banda Muerta Medida',
    'LINEALIDAD DINAMICA RECOMENDADA': 'Linealidad Recomendada',
    'LINEALIDAD DINAMICA MEDIDA': 'Linealidad Medida'
}

print("=" * 70)
print("MÉTRICAS Y GRÁFICAS COMPARATIVAS CON PASTELES")
print("=" * 70)

# ============================================================================
# 1. Cargar datos
# ============================================================================
print("\n📂 Cargando datos...")

if not os.path.exists(DATA_PATH):
    print(f"❌ Error: No existe {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"✅ Datos cargados: {len(df)} filas")

# Verificar columna de estado
if 'estado_real' not in df.columns:
    print("❌ Error: No se encontró 'estado_real' en el dataset")
    sys.exit(1)

# Definir las 8 features numéricas
features = [
    'FRICCION RECOMENDADA', 'FRICCION MEDIDA',
    'TORQUE - CARGA EN EL ASIENTO RECOMENDADA', 'TORQUE - CARGA EN EL ASIENTO MEDIDA',
    'BANDA DE ERROR DINAMICA RECOMENDADA', 'BANDA DE ERROR DINAMICA MEDIDA',
    'LINEALIDAD DINAMICA RECOMENDADA', 'LINEALIDAD DINAMICA MEDIDA'
]

# Verificar que las features existen
for f in features:
    if f not in df.columns:
        print(f"⚠️ Advertencia: Feature '{f}' no encontrada")

X = df[[f for f in features if f in df.columns]]
y_true = df['estado_real']

# Limpiar espacios en blanco en los nombres de las clases
y_true = y_true.str.strip() if hasattr(y_true, 'str') else y_true

print(f"\n📊 Distribución de clases reales (Diagnóstico Experto):")
print(y_true.value_counts())

# ============================================================================
# 2. GRÁFICA 1: Distribución de estados - BARRAS
# ============================================================================
print("\n📊 1. Generando gráfica de barras...")

fig, ax = plt.subplots(figsize=(10, 6))
counts = y_true.value_counts()
colors_bar = [COLORES_ESTADOS.get(c, '#999') for c in counts.index]
bars = ax.bar(range(len(counts)), counts.values, color=colors_bar, edgecolor='black', linewidth=1)
ax.set_xticks(range(len(counts)))
ax.set_xticklabels(counts.index, rotation=15, ha='right')
ax.set_title('Distribución de Estados - Diagnóstico Experto', fontweight='bold')
ax.set_ylabel('Cantidad de Válvulas')

for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            str(val), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_distribucion_barras.png'), dpi=300)
plt.close()
print(f"   ✅ Guardada: 01_distribucion_barras.png")

# ============================================================================
# 3. GRÁFICA 2: Distribución de estados - PASTEL (Experto)
# ============================================================================
print("\n📊 2. Generando gráfica de pastel (Experto)...")

fig, ax = plt.subplots(figsize=(8, 8))
counts = y_true.value_counts()
colors_pie = [COLORES_ESTADOS.get(c, '#999') for c in counts.index]

wedges, texts, autotexts = ax.pie(counts.values, labels=counts.index, colors=colors_pie,
                                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 12})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax.set_title('Diagnóstico Experto - Distribución de Estados', fontweight='bold')
ax.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_distribucion_pastel_experto.png'), dpi=300)
plt.close()
print(f"   ✅ Guardada: 02_distribucion_pastel_experto.png")

# ============================================================================
# 4. Función para cargar modelos
# ============================================================================
def cargar_modelo(path, nombre):
    """Carga un modelo desde archivo pickle"""
    if not os.path.exists(path):
        print(f"   ⚠️ No existe: {path}")
        return None, None, None
    
    try:
        data = joblib.load(path)
        
        if isinstance(data, dict):
            modelo = data.get('model')
            scaler = data.get('scaler')
            label_encoder = data.get('label_encoder')
            features_guardadas = data.get('features', [])
            print(f"   ✅ {nombre} cargado (formato diccionario)")
            return modelo, scaler, label_encoder
        else:
            print(f"   ⚠️ {nombre}: formato no reconocido")
            return None, None, None
            
    except Exception as e:
        print(f"   ❌ Error cargando {nombre}: {e}")
        return None, None, None

# ============================================================================
# 5. Cargar modelos
# ============================================================================
print("\n📂 Cargando modelos...")

rf_model, rf_scaler, rf_encoder = cargar_modelo(MODEL_RF_PATH, "Random Forest")
xgb_model, xgb_scaler, xgb_encoder = cargar_modelo(MODEL_XGB_PATH, "XGBoost")

# ============================================================================
# 6. Hacer predicciones
# ============================================================================
print("\n📊 Haciendo predicciones...")

resultados = {}
modelos = {}

# Random Forest predictions
if rf_model is not None:
    try:
        if rf_scaler is not None:
            X_scaled = rf_scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        
        y_pred_rf = rf_model.predict(X_scaled)
        
        # Decodificar si es necesario
        if rf_encoder is not None and isinstance(y_pred_rf[0], (int, np.integer)):
            y_pred_rf = rf_encoder.inverse_transform(y_pred_rf)
        
        resultados['Random Forest'] = y_pred_rf
        modelos['Random Forest'] = rf_model
        print("   ✅ Random Forest predicciones listas")
    except Exception as e:
        print(f"   ❌ Error en predicciones RF: {e}")

# XGBoost predictions
if xgb_model is not None:
    try:
        if xgb_scaler is not None:
            X_scaled = xgb_scaler.transform(X)
        else:
            from sklearn.preprocessing import StandardScaler
            scaler_temp = StandardScaler()
            X_scaled = scaler_temp.fit_transform(X)
        
        y_pred_xgb = xgb_model.predict(X_scaled)
        
        # Decodificar si es necesario
        if xgb_encoder is not None and isinstance(y_pred_xgb[0], (int, np.integer)):
            y_pred_xgb = xgb_encoder.inverse_transform(y_pred_xgb)
        
        resultados['XGBoost'] = y_pred_xgb
        modelos['XGBoost'] = xgb_model
        print("   ✅ XGBoost predicciones listas")
    except Exception as e:
        print(f"   ❌ Error en predicciones XGB: {e}")

if len(resultados) == 0:
    print("\n❌ ERROR: No se pudo cargar ningún modelo")
    sys.exit(1)

# ============================================================================
# 7. GRÁFICA 3: COMPARACIÓN DE PASTELES (3 en 1)
# ============================================================================
print("\n📊 3. Generando gráfica comparativa de pasteles...")

# Preparar datos para los 3 pasteles
datos_pasteles = [('Diagnóstico Experto', y_true)]
for nombre, pred in resultados.items():
    datos_pasteles.append((nombre, pred))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, (titulo, datos) in enumerate(datos_pasteles):
    # Contar valores para cada clase
    counts = {}
    for estado in ['ACEPTABLE', 'ACEPTABLE CON COMENTARIOS', 'ALERTA']:
        counts[estado] = (datos == estado).sum()
    
    # Filtrar solo clases con valores > 0
    non_zero = [(estado, counts[estado]) for estado in ['ACEPTABLE', 'ACEPTABLE CON COMENTARIOS', 'ALERTA'] if counts[estado] > 0]
    
    if len(non_zero) > 0:
        labels = [item[0] for item in non_zero]
        values = [item[1] for item in non_zero]
        colors_pie = [COLORES_ESTADOS[lab] for lab in labels]
        
        wedges, texts, autotexts = axes[idx].pie(values, labels=labels, colors=colors_pie,
                                                   autopct='%1.1f%%', startangle=90,
                                                   textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        axes[idx].text(0.5, 0.5, 'Sin datos', ha='center', va='center', fontsize=14)
    
    axes[idx].set_title(titulo, fontsize=14, fontweight='bold')
    axes[idx].axis('equal')

plt.suptitle('Comparación de Distribución de Estados', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_comparacion_pasteles.png'), dpi=300)
plt.close()
print(f"   ✅ Guardada: 03_comparacion_pasteles.png")

# ============================================================================
# 8. Métricas de los modelos
# ============================================================================
print("\n📊 Calculando métricas de los modelos...")

# Codificar etiquetas para métricas
le = LabelEncoder()
y_true_encoded = le.fit_transform(y_true)
clases = le.classes_

metricas_dict = {}
metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for nombre, y_pred in resultados.items():
    # Codificar predicciones si son strings
    if isinstance(y_pred[0], str):
        y_pred_encoded = le.transform(y_pred)
    else:
        y_pred_encoded = y_pred
    
    acc = accuracy_score(y_true_encoded, y_pred_encoded)
    prec = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
    rec = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
    f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
    
    metricas_dict[nombre] = [acc, prec, rec, f1]
    
    print(f"\n📊 {nombre}:")
    print(f"   Accuracy: {acc:.2%}")
    print(f"   Precision: {prec:.2%}")
    print(f"   Recall: {rec:.2%}")
    print(f"   F1-Score: {f1:.2%}")

# ============================================================================
# 9. GRÁFICA 4: Comparación de métricas
# ============================================================================
print("\n📊 4. Generando gráfica de comparación de métricas...")

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metricas_nombres))
width = 0.35

for i, (nombre, valores) in enumerate(metricas_dict.items()):
    offset = (i - len(metricas_dict)/2 + 0.5) * width
    color = COLORES_MODELOS.get(nombre, '#999')
    bars = ax.bar(x + offset, valores, width, label=nombre, color=color, edgecolor='black')
    
    for bar, val in zip(bars, valores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('Puntuación')
ax.set_title('Comparación de Métricas de Desempeño', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metricas_nombres)
ax.legend(loc='lower right')
ax.set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_comparacion_metricas.png'), dpi=300)
plt.close()
print(f"   ✅ Guardada: 04_comparacion_metricas.png")

# ============================================================================
# 10. GRÁFICA 5: Matrices de confusión
# ============================================================================
print("\n📊 5. Generando matrices de confusión...")

n_models = len(resultados)
fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
if n_models == 1:
    axes = [axes]

for i, (nombre, y_pred) in enumerate(resultados.items()):
    # Codificar predicciones
    if isinstance(y_pred[0], str):
        y_pred_encoded = le.transform(y_pred)
    else:
        y_pred_encoded = y_pred
    
    cm = confusion_matrix(y_true_encoded, y_pred_encoded)
    cmap = 'Blues' if 'Random' in nombre else 'Oranges'
    
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=clases, yticklabels=clases, ax=axes[i],
                annot_kws={'size': 12, 'weight': 'bold'})
    axes[i].set_title(f'{nombre} - Matriz de Confusión', fontweight='bold')
    axes[i].set_xlabel('Predicción')
    axes[i].set_ylabel('Real')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_matrices_confusion.png'), dpi=300)
plt.close()
print(f"   ✅ Guardada: 05_matrices_confusion.png")

# ============================================================================
# 11. GRÁFICA 6: Importancia de features - Random Forest
# ============================================================================
if 'Random Forest' in modelos:
    print("\n📊 6. Generando importancia de features (Random Forest)...")
    try:
        rf_model_obj = modelos['Random Forest']
        importancias = rf_model_obj.feature_importances_
        
        feature_names_legibles = [FEATURE_NAMES.get(f, f) for f in features if f in df.columns]
        importancias_clean = importancias[:len(feature_names_legibles)]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_legibles,
            'importance': importancias_clean
        }).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(feature_importance_df)))
        bars = ax.barh(feature_importance_df['feature'], feature_importance_df['importance'],
                       color=colors, edgecolor='black')
        ax.set_xlabel('Importancia')
        ax.set_title('Random Forest - Importancia de Características', fontweight='bold')
        
        for bar, imp in zip(bars, feature_importance_df['importance']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '06_feature_importance_rf.png'), dpi=300)
        plt.close()
        print(f"   ✅ Guardada: 06_feature_importance_rf.png")
    except Exception as e:
        print(f"   ⚠️ No se pudo generar: {e}")

# ============================================================================
# 12. GRÁFICA 7: Importancia de features - XGBoost
# ============================================================================
if 'XGBoost' in modelos:
    print("\n📊 7. Generando importancia de features (XGBoost)...")
    try:
        xgb_model_obj = modelos['XGBoost']
        importancias = xgb_model_obj.feature_importances_
        
        feature_names_legibles = [FEATURE_NAMES.get(f, f) for f in features if f in df.columns]
        importancias_clean = importancias[:len(feature_names_legibles)]
        
        # Normalizar para mejor visualización
        max_imp = max(importancias_clean) if len(importancias_clean) > 0 else 1
        importancias_norm = [i / max_imp for i in importancias_clean]
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_legibles,
            'importance': importancias_norm
        }).sort_values('importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.Oranges(np.linspace(0.3, 0.8, len(feature_importance_df)))
        bars = ax.barh(feature_importance_df['feature'], feature_importance_df['importance'],
                       color=colors, edgecolor='black')
        ax.set_xlabel('Importancia (normalizada)')
        ax.set_title('XGBoost - Importancia de Características', fontweight='bold')
        
        for bar, imp in zip(bars, feature_importance_df['importance']):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', va='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '07_feature_importance_xgb.png'), dpi=300)
        plt.close()
        print(f"   ✅ Guardada: 07_feature_importance_xgb.png")
    except Exception as e:
        print(f"   ⚠️ No se pudo generar: {e}")

# ============================================================================
# 13. GRÁFICA 8: Comparación de importancia de features
# ============================================================================
if 'Random Forest' in modelos and 'XGBoost' in modelos:
    print("\n📊 8. Generando comparación de importancia...")
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        feature_names_simple = ['Fricción\nRecomendada', 'Fricción\nMedida', 
                                'Torque\nRecomendado', 'Torque\nMedido',
                                'Banda\nRecomendada', 'Banda\nMedida',
                                'Linealidad\nRecomendada', 'Linealidad\nMedida']
        
        # Random Forest
        imp_rf = modelos['Random Forest'].feature_importances_[:len(feature_names_simple)]
        indices_rf = np.argsort(imp_rf)
        axes[0].barh([feature_names_simple[i] for i in indices_rf], imp_rf[indices_rf],
                     color=COLORES_MODELOS['Random Forest'], edgecolor='black')
        axes[0].set_title('Random Forest', fontweight='bold')
        axes[0].set_xlabel('Importancia')
        
        # XGBoost
        imp_xgb = modelos['XGBoost'].feature_importances_[:len(feature_names_simple)]
        imp_xgb_norm = imp_xgb / max(imp_xgb) if max(imp_xgb) > 0 else imp_xgb
        indices_xgb = np.argsort(imp_xgb_norm)
        axes[1].barh([feature_names_simple[i] for i in indices_xgb], imp_xgb_norm[indices_xgb],
                     color=COLORES_MODELOS['XGBoost'], edgecolor='black')
        axes[1].set_title('XGBoost', fontweight='bold')
        axes[1].set_xlabel('Importancia (normalizada)')
        
        fig.suptitle('Comparación de Importancia de Características', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '08_comparacion_feature_importance.png'), dpi=300)
        plt.close()
        print(f"   ✅ Guardada: 08_comparacion_feature_importance.png")
    except Exception as e:
        print(f"   ⚠️ No se pudo generar: {e}")

# ============================================================================
# 14. GRÁFICA 9: Comparación predicciones vs real (XGBoost)
# ============================================================================
if 'XGBoost' in resultados:
    print("\n📊 9. Generando comparación real vs predicción...")
    try:
        y_pred_xgb = resultados['XGBoost']
        if isinstance(y_pred_xgb[0], (int, np.integer)):
            y_pred_xgb_str = le.inverse_transform(y_pred_xgb)
        else:
            y_pred_xgb_str = y_pred_xgb
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cm_comp = confusion_matrix(y_true, y_pred_xgb_str, labels=clases)
        
        sns.heatmap(cm_comp, annot=True, fmt='d', cmap='Greens',
                    xticklabels=clases, yticklabels=clases, ax=ax,
                    annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title('XGBoost: Comparación Real vs Predicción', fontweight='bold')
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real (Diagnóstico Experto)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '09_comparacion_real_vs_predicho.png'), dpi=300)
        plt.close()
        print(f"   ✅ Guardada: 09_comparacion_real_vs_predicho.png")
    except Exception as e:
        print(f"   ⚠️ No se pudo generar: {e}")

# ============================================================================
# 15. Tabla resumen de métricas
# ============================================================================
print("\n" + "=" * 70)
print("TABLA RESUMEN DE MÉTRICAS")
print("=" * 70)

print("\n┌─────────────────────┬─────────────────┬─────────────────┐")
print("│       Métrica       │  Random Forest  │     XGBoost     │")
print("├─────────────────────┼─────────────────┼─────────────────┤")

for i, metrica in enumerate(metricas_nombres):
    rf_val = metricas_dict.get('Random Forest', [0, 0, 0, 0])[i] if 'Random Forest' in metricas_dict else 0
    xgb_val = metricas_dict.get('XGBoost', [0, 0, 0, 0])[i] if 'XGBoost' in metricas_dict else 0
    print(f"│ {metrica:<19} │     {rf_val:.3f}     │     {xgb_val:.3f}     │")

print("└─────────────────────┴─────────────────┴─────────────────┘")

if 'Random Forest' in metricas_dict and 'XGBoost' in metricas_dict:
    mejor = "XGBoost" if metricas_dict['XGBoost'][0] > metricas_dict['Random Forest'][0] else "Random Forest"
    diferencia = abs(metricas_dict['XGBoost'][0] - metricas_dict['Random Forest'][0])
    print(f"\n✅ Mejor modelo en accuracy: {mejor} (diferencia de {diferencia:.3f})")

# ============================================================================
# 16. Tabla de distribución de estados
# ============================================================================
print("\n" + "=" * 70)
print("COMPARACIÓN DE DISTRIBUCIÓN DE ESTADOS")
print("=" * 70)

print("\n┌─────────────────────────┬─────────────────┬─────────────────┬─────────────────┐")
print("│         Estado          │    Experto      │  Random Forest  │    XGBoost      │")
print("├─────────────────────────┼─────────────────┼─────────────────┼─────────────────┤")

for estado in ['ACEPTABLE', 'ACEPTABLE CON COMENTARIOS', 'ALERTA']:
    exp_cnt = (y_true == estado).sum()
    exp_pct = exp_cnt / len(y_true) * 100
    
    rf_cnt = (resultados.get('Random Forest', pd.Series([])) == estado).sum() if 'Random Forest' in resultados else 0
    rf_pct = rf_cnt / len(y_true) * 100 if len(y_true) > 0 else 0
    
    xgb_cnt = (resultados.get('XGBoost', pd.Series([])) == estado).sum() if 'XGBoost' in resultados else 0
    xgb_pct = xgb_cnt / len(y_true) * 100 if len(y_true) > 0 else 0
    
    print(f"│ {estado:<23} │ {exp_cnt:>3} ({exp_pct:>5.1f}%) │ {rf_cnt:>3} ({rf_pct:>5.1f}%) │ {xgb_cnt:>3} ({xgb_pct:>5.1f}%) │")

print("└─────────────────────────┴─────────────────┴─────────────────┴─────────────────┘")

# ============================================================================
# 17. Resumen final
# ============================================================================
print("\n" + "=" * 70)
print("✅ GRÁFICAS Y MÉTRICAS COMPLETADAS")
print(f"📁 Gráficas guardadas en: {OUTPUT_DIR}")
print("\n📊 Gráficas generadas:")
print("   1. 01_distribucion_barras.png")
print("   2. 02_distribucion_pastel_experto.png")
print("   3. 03_comparacion_pasteles.png ⭐ (3 pasteles lado a lado)")
print("   4. 04_comparacion_metricas.png")
print("   5. 05_matrices_confusion.png")
print("   6. 06_feature_importance_rf.png")
print("   7. 07_feature_importance_xgb.png")
print("   8. 08_comparacion_feature_importance.png")
print("   9. 09_comparacion_real_vs_predicho.png")
print("=" * 70)