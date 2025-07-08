
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Creando datos

np.random.seed(42)
fechas = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n_productos = ['Laptop', 'Mouse', 'Teclado', 'Monitor', 'Auriculares']
regiones =['Norte', 'Sur', 'Este', 'Oeste', 'Centro']

# Generando el dataset

data = []
for fecha in fechas:
    for _ in range(np.random.poisson(10)): # 10 Ventas promedio por día
        data.append({
            'Fecha': fecha,
            'Producto': np.random.choice(n_productos),
            'Region': np.random.choice(regiones),
            'cantidad': np.random.randint(1, 10),
            'precio_unitario': np.random.uniform(50, 1500),
            'vendedor': f'Vendedor_{np.random.randint(1, 21)}'
        })

df = pd.DataFrame(data)

# Vamos a gregar una nueva columna de ventas totales

df['venta_total'] = df['cantidad'] * df['precio_unitario']

# Vamos a conocer mas del dataframe

print("Shape del datase:", df.shape)
print("\nPrimeras filas:")
print(df.head())
print('\nInformación general')
print(df.info())
print('\nDescripción estadística')
print(df.describe())

#  1. Organizar las ventas por mes

df_monthly = df.groupby(df['Fecha'].dt.to_period('M'))['venta_total'].sum().reset_index()
df_monthly['fecha'] = df_monthly['Fecha'].astype(str)


fig_monthly  = px.line(df_monthly, x='fecha', y='venta_total',
                       title='Tendencia de ventas mensuales',
                       labels={'venta_total': 'Ventas ($)', 'fecha': 'Mes'})
fig_monthly.update_traces(line=dict(width=3))


# 2. Top productos

df_productos = df.groupby('Producto')['venta_total'].sum().sort_values(ascending=True)
fig_productos = px.bar(x=df_productos.values, y=df_productos.index,
                      orientation='h',
                      title='Top productos por ventas',
                      labels={'x': 'Ventas ($)', 'y': 'Producto'})

# 3. Análisis geografico

df_regiones = df.groupby('Region')['venta_total'].sum().reset_index()
fig_regiones = px.pie(df_regiones, values='venta_total', names='Region',
                     title='Distribución de ventas por región',
                     labels={'venta_total': 'Ventas ($)'})

# 4. Correlación entre variables

df_corr = df[['cantidad', 'precio_unitario', 'venta_total']].corr()
fig_heamap = px.imshow(df_corr, text_auto=True, aspect='auto',
                       title='Correlación entre variables',
                       labels=dict(x='Variables', y='Variables'))

# 5. Distribución de ventas

fig_dist = px.histogram(df, x='venta_total', nbins=50,
                        title='Distribución de ventas individuales',
                        labels={'venta_total': 'Ventas ($)'})

# Configuración de la pagina

st.set_page_config(page_title='Dashboard de ventas',
                   page_icon=':bar_chart:',
                   layout='wide')

# Titulo principal

st.title('Dashboard de ventas')
st.markdown('---')

# Slider para filtros

st.sidebar.header('Filtros')
productos_seleccionados = st.sidebar.multiselect(
    'Seleccionar productos',
    options=df['Producto'].unique(),
    default=df['Producto'].unique()
)

# Filtro por región

regiones_seleccionadas = st.sidebar.multiselect(
    'Seleccionar regiones',
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

# Filtrar objetos basados en seleccion

df_filtrado = df[
    (df['Producto'].isin(productos_seleccionados)) &
    (df['Region'].isin(regiones_seleccionadas))
]


# Metricas principales

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric('Ventas totales', f'${df_filtrado["venta_total"].sum():.0f}')

with col2:
    st.metric('Ventas promedio', f'${df_filtrado["venta_total"].mean():.2f}')
with col3:
    st.metric('Ventas maximas', f'${len(df_filtrado):,}')
with col4:
    # Calculate annual growth on the original dataframe
    ventas_2023 = df[df['Fecha'].dt.year == 2023]['venta_total'].sum()
    ventas_2024 = df[df['Fecha'].dt.year == 2024]['venta_total'].sum()
    crecimiento = ((ventas_2024 / ventas_2023) - 1) * 100
    st.metric('Crecimiento anual', f'{crecimiento:.1f}%')
    
    
# Layout con dos columnas

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_monthly, use_container_width=True)
    st.plotly_chart(fig_productos, use_container_width=True)
with col2:
    st.plotly_chart(fig_regiones, use_container_width=True)
    st.plotly_chart(fig_heamap, use_container_width=True)

# Grafio completo en la parte inferior

st.plotly_chart(fig_dist, use_container_width=True)


