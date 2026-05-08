# Importar librerías necesarias
import pandas as pd #Libreria para trabajar con tablas de datos
import numpy as np #Libreria para oepraciones matematicas t arrays numericos con el apodo np
import matplotlib.pyplot as plt #Libreria para realizar graficos basicos con el apodo plt
import seaborn as sns # Libreria para graficos mas esteticos y estadisticos con el apodo sns

# Configurar visualización
plt.style.use('seaborn-v0_8-darkgrid') #Se define el estilo visual de los graficos
sns.set_palette("Set2") # Se define la paleta de colores que usaran los graficos
plt.rcParams['figure.figsize'] = (12, 6)# Se define tamaño de graficos

print("Librerías importadas correctamente") #Muestra mensaje en pantalla para confirmar la improtacion de librerias correctamente
print(f"Pandas versión: {pd.__version__}")#Muestra la version de pandas instalado