import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diagnóstico de Válvulas", layout="wide")
st.title("🔧 Sistema de Diagnóstico de Válvulas de Control")
st.markdown("*Automatización de informes técnicos con Machine Learning*")
st.markdown("---")

# Datos de ejemplo
np.random.seed(42)
n = 100

datos = {
    'TAG': [f'VAL-{i:03d}' for i in range(n)],
    'FRICCION_REC': np.random.uniform(50, 150, n),
    'FRICCION_MED': np.random.uniform(40, 160, n),
    'CARGA_REC': np.random.uniform(100, 500, n),
    'CARGA_MED': np.random.uniform(80, 600, n),
    'BANDA_REC': np.random.uniform(2, 8, n),
    'BANDA_MED': np.random.uniform(1, 12, n),
    'LINEALIDAD_REC': np.random.uniform(1, 5, n),
    'LINEALIDAD_MED': np.random.uniform(0.5, 7, n)
}

df = pd.DataFrame(datos)

# Calcular puntuaciones
df['punt_friccion'] = df.apply(lambda x: 1 if abs(x['FRICCION_MED'] - x['FRICCION_REC']) <= 5 else
                                2 if abs(x['FRICCION_MED'] - x['FRICCION_REC']) <= 10 else 3, axis=1)

df['punt_carga'] = df.apply(lambda x: 1 if abs((x['CARGA_MED']/x['CARGA_REC']-1)*100) <= 10 else
                             2 if abs((x['CARGA_MED']/x['CARGA_REC']-1)*100) <= 25 else 3, axis=1)

df['punt_banda'] = df.apply(lambda x: 1 if x['BANDA_MED'] <= x['BANDA_REC'] else
                             2 if x['BANDA_MED'] <= x['BANDA_REC'] * 1.2 else 3, axis=1)

df['punt_linealidad'] = df.apply(lambda x: 1 if x['LINEALIDAD_MED'] <= x['LINEALIDAD_REC'] else
                                  2 if x['LINEALIDAD_MED'] <= x['LINEALIDAD_REC'] * 1.3 else 3, axis=1)

df['puntuacion_total'] = df['punt_friccion'] + df['punt_carga'] + df['punt_banda'] + df['punt_linealidad']

def get_estado(p):
    if p <= 4:
        return "ACEPTABLE (VERDE)"
    elif p <= 7:
        return "ACEPTABLE CON COMENTARIOS (AMARILLO)"
    else:
        return "ALERTA (NARANJA)"

df['estado'] = df['puntuacion_total'].apply(get_estado)

# Métricas
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Válvulas", len(df))
col2.metric("VERDE", len(df[df['estado'] == "ACEPTABLE (VERDE)"]))
col3.metric("AMARILLO", len(df[df['estado'] == "ACEPTABLE CON COMENTARIOS (AMARILLO)"]))
col4.metric("NARANJA", len(df[df['estado'] == "ALERTA (NARANJA)"]))

st.markdown("---")

# Gráfico
st.subheader("📊 Distribución de Estados")
fig, ax = plt.subplots(figsize=(10, 6))
df['estado'].value_counts().plot(kind='bar', ax=ax, color=['green', 'orange', 'red'], edgecolor='black')
ax.set_xlabel("Estado")
ax.set_ylabel("Número de Válvulas")
ax.set_title("Estado de las Válvulas")
for i, v in enumerate(df['estado'].value_counts().values):
    ax.text(i, v + 2, str(v), ha='center', fontweight='bold')
st.pyplot(fig)

st.markdown("---")

# Tabla de alertas
st.subheader("🚨 Válvulas en estado ALERTA (NARANJA)")
alertas = df[df['estado'] == "ALERTA (NARANJA)"]
if len(alertas) > 0:
    st.dataframe(alertas[['TAG', 'punt_friccion', 'punt_carga', 'punt_banda', 'punt_linealidad', 'puntuacion_total']])
else:
    st.success("✅ No hay válvulas en estado ALERTA")

st.markdown("---")
st.caption("TFM - Diagnóstico de Válvulas de Control")