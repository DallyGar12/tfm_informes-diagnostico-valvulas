import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Válvulas de Control",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título
st.title("🔧 Sistema de Diagnóstico de Válvulas de Control")
st.markdown("*Automatización de informes técnicos con Machine Learning*")
st.markdown("---")

# ============================================================================
# CARGA DE DATOS
# ============================================================================

@st.cache_data
def cargar_datos():
    """Cargar el dataset de 200 válvulas"""
    try:
        # Intenta cargar tu CSV real
        df = pd.read_csv("C:/Users/dally/OneDrive - UNIR/TRABAJO FIN DE MASTER/tfm_informes-diagnostico-valvulas/Dataset valvulas de control SK.csv", sep=';', encoding='utf-8')
        st.success(f"✅ Datos cargados correctamente: {len(df)} válvulas")
        return df
    except:
        # Si no existe, crear datos de ejemplo (200 válvulas)
        st.warning("⚠️ No se encontró el archivo. Usando datos de ejemplo.")
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'TAG': [f'VAL-{i:03d}' for i in range(n)],
            'FRICCION_REC': np.random.uniform(50, 150, n).round(1),
            'FRICCION_MED': np.random.uniform(40, 160, n).round(1),
            'CARGA_REC': np.random.uniform(100, 500, n).round(1),
            'CARGA_MED': np.random.uniform(80, 600, n).round(1),
            'BANDA_REC': np.random.uniform(2, 8, n).round(2),
            'BANDA_MED': np.random.uniform(1, 12, n).round(2),
            'LINEALIDAD_REC': np.random.uniform(1, 5, n).round(2),
            'LINEALIDAD_MED': np.random.uniform(0.5, 7, n).round(2)
        })
        return df

df = cargar_datos()

# ============================================================================
# CÁLCULO DE PUNTUACIONES Y DIAGNÓSTICOS
# ============================================================================

# 1. Puntuación de FRICCIÓN
def calc_punt_friccion(rec, med):
    desviacion = abs(med - rec)
    if desviacion <= 5:
        return 1, "Normal", "Fricción dentro de especificaciones"
    elif desviacion <= 10:
        return 2, "Moderada", "Revisar packing: posible falta de lubricación o desgaste"
    else:
        return 3, "Severa", "Inspección urgente de packing. Riesgo de fuga o atascamiento"

# 2. Puntuación de CARGA EN ASIENTO
def calc_punt_carga(rec, med):
    if rec <= 0:
        return 0, "Error", "Valor recomendado inválido"
    porcentaje = (med / rec - 1) * 100
    if abs(porcentaje) <= 10:
        return 1, "Normal", "Carga dentro de especificaciones. Sellado correcto"
    elif porcentaje < -10:
        return 2, "Moderada (baja)", "Verificar estado de internos (asiento y tapón). Posible desgaste"
    elif porcentaje > 15:
        return 2, "Moderada (alta)", "Monitorear tendencia. La sobrecarga puede generar desgaste prematuro"
    elif porcentaje < -15:
        return 3, "Severa (baja)", "Inspección urgente. Probable fuga interna por desgaste severo"
    else:
        return 3, "Severa (alta)", "Riesgo de daños prematuros en asiento y tapón"

# 3. Puntuación de BANDA MUERTA
def calc_punt_banda(rec, med):
    if rec <= 0:
        return 0, "Error", "Valor recomendado inválido"
    if med <= rec:
        return 1, "Normal", "Banda muerta dentro de especificaciones"
    elif med <= rec * 1.2:
        return 2, "Moderada", "Verificar calibración del posicionador"
    else:
        return 3, "Severa", "Revisar urgentemente el posicionador"

# 4. Puntuación de LINEALIDAD DINÁMICA
def calc_punt_linealidad(rec, med):
    if rec <= 0:
        return 0, "Error", "Valor recomendado inválido"
    if med <= rec:
        return 1, "Normal", "Linealidad dentro de especificaciones"
    elif med <= rec * 1.3:
        return 2, "Moderada", "Verificar caracterización del posicionador"
    else:
        return 3, "Severa", "Revisar urgentemente el posicionador y su acople"

# Aplicar cálculos a todas las válvulas
resultados = []
for idx, row in df.iterrows():
    f_punt, f_nivel, f_recom = calc_punt_friccion(row['FRICCION_REC'], row['FRICCION_MED'])
    c_punt, c_nivel, c_recom = calc_punt_carga(row['CARGA_REC'], row['CARGA_MED'])
    b_punt, b_nivel, b_recom = calc_punt_banda(row['BANDA_REC'], row['BANDA_MED'])
    l_punt, l_nivel, l_recom = calc_punt_linealidad(row['LINEALIDAD_REC'], row['LINEALIDAD_MED'])
    
    punt_total = f_punt + c_punt + b_punt + l_punt
    
    # Estado global
    if punt_total <= 4:
        estado = "🟢 ACEPTABLE"
        color = "green"
        accion = "Monitoreo rutinario"
    elif punt_total <= 7:
        estado = "🟡 ACEPTABLE CON COMENTARIOS"
        color = "orange"
        accion = "Planificar mantenimiento preventivo"
    else:
        estado = "🔴 ALERTA"
        color = "red"
        accion = "Intervención prioritaria requerida"
    
    resultados.append({
        'TAG': row['TAG'],
        'f_punt': f_punt, 'f_nivel': f_nivel, 'f_recom': f_recom,
        'c_punt': c_punt, 'c_nivel': c_nivel, 'c_recom': c_recom,
        'b_punt': b_punt, 'b_nivel': b_nivel, 'b_recom': b_recom,
        'l_punt': l_punt, 'l_nivel': l_nivel, 'l_recom': l_recom,
        'punt_total': punt_total,
        'estado': estado, 'color': color, 'accion': accion
    })

df_resultados = pd.DataFrame(resultados)
df_original = df.copy()
df = df_resultados

# ============================================================================
# BARRA LATERAL - FILTROS
# ============================================================================

st.sidebar.header("🔍 Filtros")

# Filtro por estado
estados = st.sidebar.multiselect(
    "Estado de la válvula",
    options=df['estado'].unique(),
    default=df['estado'].unique()
)

# Filtro por puntuación
min_punt = st.sidebar.slider("Puntuación mínima", 0, 12, 0)
max_punt = st.sidebar.slider("Puntuación máxima", 0, 12, 12)

# Aplicar filtros
df_filtrado = df[
    (df['estado'].isin(estados)) &
    (df['punt_total'] >= min_punt) &
    (df['punt_total'] <= max_punt)
]

st.sidebar.markdown("---")
st.sidebar.metric("📊 Válvulas filtradas", len(df_filtrado))
st.sidebar.metric("📋 Total en dataset", len(df))

# ============================================================================
# MÉTRICAS PRINCIPALES
# ============================================================================

st.subheader("📊 Panel de Control")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Válvulas", len(df))
col2.metric("🟢 ACEPTABLE", len(df[df['estado'] == "🟢 ACEPTABLE"]))
col3.metric("🟡 ACEPTABLE CON COMENTARIOS", len(df[df['estado'] == "🟡 ACEPTABLE CON COMENTARIOS"]))
col4.metric("🔴 ALERTA", len(df[df['estado'] == "🔴 ALERTA"]))
col5.metric("Puntuación Promedio", f"{df['punt_total'].mean():.1f}")

st.markdown("---")

# ============================================================================
# GRÁFICOS DE ANÁLISIS
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Distribución de Estados")
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df['estado'].value_counts()
    colors = {'🟢 ACEPTABLE': 'green', '🟡 ACEPTABLE CON COMENTARIOS': 'orange', '🔴 ALERTA': 'red'}
    bar_colors = [colors.get(x, 'gray') for x in counts.index]
    counts.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black')
    ax.set_xlabel("Estado")
    ax.set_ylabel("Número de Válvulas")
    ax.set_title("Distribución de Estados de las Válvulas")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
    st.pyplot(fig)

with col2:
    st.subheader("📈 Distribución de Puntuaciones")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['punt_total'], bins=12, color='#3498DB', edgecolor='black', alpha=0.7)
    ax.axvline(x=4, color='green', linestyle='--', linewidth=2, label='Límite ACEPTABLE')
    ax.axvline(x=7, color='orange', linestyle='--', linewidth=2, label='Límite MODERADA')
    ax.set_xlabel("Puntuación Total")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de Puntuaciones de Diagnóstico")
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# ============================================================================
# GRÁFICOS POR PARÁMETRO
# ============================================================================

st.subheader("📊 Análisis por Parámetro")

col1, col2 = st.columns(2)

with col1:
    # Gráfico de Fricción
    st.markdown("**🔧 Fricción**")
    fig, ax = plt.subplots(figsize=(8, 4))
    friccion_counts = df['f_nivel'].value_counts()
    colores_nivel = {'Normal': 'green', 'Moderada': 'orange', 'Severa': 'red'}
    bar_colors_f = [colores_nivel.get(x, 'gray') for x in friccion_counts.index]
    friccion_counts.plot(kind='bar', ax=ax, color=bar_colors_f, edgecolor='black')
    ax.set_title("Niveles de Fricción")
    ax.set_xlabel("Nivel")
    ax.set_ylabel("Válvulas")
    st.pyplot(fig)
    
    # Gráfico de Banda Muerta
    st.markdown("**📡 Banda Muerta**")
    fig, ax = plt.subplots(figsize=(8, 4))
    banda_counts = df['b_nivel'].value_counts()
    bar_colors_b = [colores_nivel.get(x, 'gray') for x in banda_counts.index]
    banda_counts.plot(kind='bar', ax=ax, color=bar_colors_b, edgecolor='black')
    ax.set_title("Niveles de Banda Muerta")
    ax.set_xlabel("Nivel")
    ax.set_ylabel("Válvulas")
    st.pyplot(fig)

with col2:
    # Gráfico de Carga
    st.markdown("**⚙️ Carga en Asiento**")
    fig, ax = plt.subplots(figsize=(8, 4))
    carga_counts = df['c_nivel'].value_counts()
    bar_colors_c = [colores_nivel.get(x.split()[0] if ' ' in x else x, 'gray') for x in carga_counts.index]
    carga_counts.plot(kind='bar', ax=ax, color=bar_colors_c, edgecolor='black')
    ax.set_title("Niveles de Carga en Asiento")
    ax.set_xlabel("Nivel")
    ax.set_ylabel("Válvulas")
    st.pyplot(fig)
    
    # Gráfico de Linealidad
    st.markdown("**📐 Linealidad Dinámica**")
    fig, ax = plt.subplots(figsize=(8, 4))
    linealidad_counts = df['l_nivel'].value_counts()
    bar_colors_l = [colores_nivel.get(x, 'gray') for x in linealidad_counts.index]
    linealidad_counts.plot(kind='bar', ax=ax, color=bar_colors_l, edgecolor='black')
    ax.set_title("Niveles de Linealidad Dinámica")
    ax.set_xlabel("Nivel")
    ax.set_ylabel("Válvulas")
    st.pyplot(fig)

st.markdown("---")

# ============================================================================
# EXPLICACIÓN DE PUNTUACIONES
# ============================================================================

with st.expander("📖 ¿Cómo se calculan las puntuaciones? - Explicación detallada"):
    st.markdown("""
    ### 📊 Sistema de Puntuación de Diagnóstico
    
    Cada parámetro recibe una puntuación de 1 a 3 según su desviación respecto al valor recomendado:
    
    | **Parámetro** | **Puntuación 1 (Normal)** | **Puntuación 2 (Moderada)** | **Puntuación 3 (Severa)** |
    |---------------|--------------------------|----------------------------|--------------------------|
    | **Fricción** | Desviación ≤ 5 unidades | Desviación entre 5-10 unidades | Desviación > 10 unidades |
    | **Carga en Asiento** | Desviación ≤ 10% | 10-15% bajo o 15-25% alto | >15% bajo o >25% alto |
    | **Banda Muerta** | Medido ≤ recomendado | Medido > rec hasta +20% | Medido > +20% sobre rec |
    | **Linealidad** | Medido ≤ recomendado | Medido > rec hasta +30% | Medido > +30% sobre rec |
    
    ### 🎯 Puntuación Total (suma de 4 parámetros)
    
    - **4 puntos**: Todo normal → 🟢 ACEPTABLE
    - **5-7 puntos**: Desviaciones moderadas → 🟡 ACEPTABLE CON COMENTARIOS
    - **8-12 puntos**: Desviaciones severas → 🔴 ALERTA
    """)

st.markdown("---")

# ============================================================================
# TABLA DE VÁLVULAS CRÍTICAS (ALERTA)
# ============================================================================

st.subheader("🚨 VÁLVULAS EN ESTADO CRÍTICO (ALERTA)")

alertas = df[df['estado'] == "🔴 ALERTA"]

if len(alertas) > 0:
    st.warning(f"⚠️ Se han identificado {len(alertas)} válvulas en estado crítico que requieren atención inmediata.")
    
    # Mostrar tabla resumen
    tabla_alertas = alertas[['TAG', 'punt_total', 'f_nivel', 'c_nivel', 'b_nivel', 'l_nivel']].copy()
    tabla_alertas.columns = ['TAG', 'Punt. Total', 'Fricción', 'Carga', 'Banda', 'Linealidad']
    st.dataframe(tabla_alertas, use_container_width=True)
    
    # Detalle de cada válvula ALERTA
    st.markdown("### 📋 Recomendaciones detalladas por válvula")
    
    for idx, row in alertas.iterrows():
        with st.expander(f"🔴 {row['TAG']} - Puntuación Total: {row['punt_total']}"):
            st.markdown(f"""
            **Diagnóstico General:** {row['estado']} - {row['accion']}
            
            | Parámetro | Nivel | Recomendación Técnica |
            |-----------|-------|----------------------|
            | 🔧 **Fricción** | {row['f_nivel']} | {row['f_recom']} |
            | ⚙️ **Carga en Asiento** | {row['c_nivel']} | {row['c_recom']} |
            | 📡 **Banda Muerta** | {row['b_nivel']} | {row['b_recom']} |
            | 📐 **Linealidad** | {row['l_nivel']} | {row['l_recom']} |
            
            **Acción recomendada:** {row['accion']}
            """)
else:
    st.success("✅ No hay válvulas en estado crítico (ALERTA)")

st.markdown("---")

# ============================================================================
# VÁLVULAS EN ESTADO ACEPTABLE CON COMENTARIOS
# ============================================================================

st.subheader("🟡 VÁLVULAS EN ESTADO ACEPTABLE CON COMENTARIOS")

moderadas = df[df['estado'] == "🟡 ACEPTABLE CON COMENTARIOS"]

if len(moderadas) > 0:
    st.info(f"ℹ️ {len(moderadas)} válvulas requieren atención preventiva programada.")
    
    for idx, row in moderadas.head(10).iterrows():
        with st.expander(f"🟡 {row['TAG']} - Puntuación Total: {row['punt_total']}"):
            st.markdown(f"""
            | Parámetro | Nivel | Recomendación |
            |-----------|-------|---------------|
            | 🔧 Fricción | {row['f_nivel']} | {row['f_recom']} |
            | ⚙️ Carga | {row['c_nivel']} | {row['c_recom']} |
            | 📡 Banda Muerta | {row['b_nivel']} | {row['b_recom']} |
            | 📐 Linealidad | {row['l_nivel']} | {row['l_recom']} |
            
            **Acción:** {row['accion']}
            """)
    
    if len(moderadas) > 10:
        st.caption(f"Mostrando 10 de {len(moderadas)} válvulas. Usa los filtros para más detalles.")
else:
    st.success("✅ No hay válvulas en estado ACEPTABLE CON COMENTARIOS")

st.markdown("---")

# ============================================================================
# BÚSQUEDA DE VÁLVULA ESPECÍFICA
# ============================================================================

st.subheader("🔍 Buscar Válvula Específica")

tag_buscar = st.text_input("Ingrese el TAG de la válvula (ej: VAL-001)", "")

if tag_buscar:
    valvula = df[df['TAG'].str.upper() == tag_buscar.upper()]
    if len(valvula) > 0:
        row = valvula.iloc[0]
        
        color_fondo = {"🟢 ACEPTABLE": "lightgreen", "🟡 ACEPTABLE CON COMENTARIOS": "#FFF3CD", "🔴 ALERTA": "#F8D7DA"}
        st.markdown(f'<div style="padding: 20px; background-color: {color_fondo.get(row["estado"], "white")}; border-radius: 10px;">', unsafe_allow_html=True)
        
        st.markdown(f"""
        ## {row['TAG']}
        
        ### 📊 Resumen General
        | Métrica | Valor |
        |---------|-------|
        | **Estado** | {row['estado']} |
        | **Puntuación Total** | {row['punt_total']} / 12 |
        | **Acción Requerida** | {row['accion']} |
        """)
        
        st.markdown("### 🔧 Diagnóstico Detallado por Parámetro")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            **Fricción:** {row['f_nivel']}  
            *{row['f_recom']}*
            
            **Carga en Asiento:** {row['c_nivel']}  
            *{row['c_recom']}*
            """)
        with col2:
            st.markdown(f"""
            **Banda Muerta:** {row['b_nivel']}  
            *{row['b_recom']}*
            
            **Linealidad Dinámica:** {row['l_nivel']}  
            *{row['l_recom']}*
            """)
        
        st.markdown(f"**📋 Conclusión:** {row['accion']}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.error(f"No se encontró la válvula con TAG '{tag_buscar}'")

st.markdown("---")
st.caption("TFM - Sistema de Diagnóstico de Válvulas de Control | Basado en criterios técnicos y Machine Learning")