# Mostrar distribución de clases actual
import requests
import json

response = requests.get('http://localhost:5000/api/resumen')
data = response.json()

print("=" * 50)
print("DISTRIBUCIÓN DE CLASES - VÁLVULAS")
print("=" * 50)
print(f"🟢 ACEPTABLE (Verde):      {data['aceptable']} válvulas")
print(f"🟡 ACEPTABLE CON COMENTARIOS (Amarillo): {data['aceptable_com']} válvulas")
print(f"🔴 ALERTA (Naranja):      {data['alerta']} válvulas")
print(f"📊 Total:                 {data['total']} válvulas")
print("=" * 50)