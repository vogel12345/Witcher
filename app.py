import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import os

# Establecer el título de la página
st.set_page_config(page_title='The Witcher 3 Wild Hunt')

# Definir el estilo CSS personalizado para la página
st.markdown("""
    <style>
        /* Establecer el fondo en blanco */
        .st-emotion-cache-ffhzg2 {
            background-image: url("https://www.lavanguardia.com/andro4all/hero/2023/06/steam.1686140277.6997.jpg");
            background-size: cover;
        }
        /* Quitar barra */
        .st-emotion-cache-1avcm0n
        {
            visibility: hidden;
        }
        .st-emotion-cache-fis6aj{
            visibility: hidden;
            padding:0;
        }
        .st-emotion-cache-1on073z e1b2p2ww0{
            margin:0;
            padding:0;
        }
    </style>
""", unsafe_allow_html=True)

def download_csv():
    with open('steam.csv', 'rb') as f:
        csv = f.read()
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="steam.csv">Descargar archivo CSV recortado</a>'
    return href

# Definir la función para generar el enlace a Google Drive
def generate_drive_link(file_id, filename):
    base_url = 'https://drive.google.com/uc?export=download&id='
    drive_link = base_url + file_id
    href = f'<a href="{drive_link}" target="_blank">{filename}</a>'
    return href

# ID del archivo en Google Drive
file_id = '1rmjw6-pv5UlKyMvuhWm-JgJVLCF7y7cy'

# Nombre del archivo que se mostrará en el enlace
filename_drive = 'Descargar archivo CSV completo'

# Generar el enlace a Google Drive
drive_link = generate_drive_link(file_id, filename_drive)

# Crear dos columnas para colocar los botones uno al lado del otro
col1, col2 = st.columns(2)

# Colocar el botón de descarga local en la primera columna
col1.markdown(download_csv(), unsafe_allow_html=True)

# Colocar el botón de descarga de Google Drive en la segunda columna
col2.markdown(drive_link, unsafe_allow_html=True)


uploaded_file = st.file_uploader('**Selecciona el archivo**', type='csv')

if st.button("Eliminar CSV"):
    # Reiniciar el DataFrame
    df = pd.DataFrame()
    uploaded_file = 0;
    st.write("CSV eliminado. Puedes cargar otro archivo.")

def get_binary_file_downloader_html(bin_file_path, label='Archivo'):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{label}.csv">Descargar {label}</a>'
    return href

def get_download_folder():
    # Obtener la carpeta de descargas del sistema operativo
    if os.name == 'nt':  # Sistema Windows
        download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    elif os.name == 'posix':  # Sistema tipo Unix (Linux, macOS)
        download_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    else:
        # Sistema operativo desconocido, utiliza una ruta genérica
        download_folder = os.path.join(os.path.expanduser("~"), "Downloads")

    return download_folder

if uploaded_file:
    df_placeholder = st.empty()  # Marcador de posición para el DataFrame
    df = pd.read_csv(uploaded_file)

    # Mostrar ventana de datos
    df_placeholder.subheader("Datos del archivo CSV")
    df_placeholder.dataframe(df)

    # Pregunta sobre eliminar registros
    delete_records = st.radio("**¿Deseas eliminar registros?**", ("Sí", "No"))

    if delete_records == "Sí":
        # Cuadro de entrada para la cantidad de registros a eliminar
        num_records_to_delete = st.number_input("**Ingrese la cantidad de registros a eliminar:**", min_value=0, max_value=len(df), step=1)

        # Botón para eliminar registros
        if st.button("Eliminar Registros"):
            # Seleccionar una muestra aleatoria de filas
            random_sample = df.sample(n=num_records_to_delete, random_state=42)

            # Eliminar las filas seleccionadas
            df = df.drop(random_sample.index)

            # Mensaje de éxito solo si se eliminaron registros
            if num_records_to_delete > 0:
                st.success(f"Se eliminaron {num_records_to_delete} registros al azar.")
            else:
                st.info("No se eliminaron registros.")

            # Guardar el DataFrame actualizado en un nuevo archivo CSV
            updated_csv_path = "archivo_actualizado.csv"
            df.to_csv(updated_csv_path, index=False)

            # Botón para descargar el CSV actualizado
            if st.download_button("Descargar CSV Actualizado", key="download_button", data=df.to_csv(index=False), file_name="archivo_actualizado.csv"):
                st.success("¡Descarga iniciada!")
        # Evitar mostrar el botón de descarga CSV si la cantidad inicial no es válida
        if "download_button" not in st.session_state:
        st.info("Carga un archivo válido para activar la descarga del CSV actualizado.")
    else:
        # Limpiar la pantalla
        df_placeholder.text('')
        # Convierte las fechas a objetos datetime
        df['timestamp_created'] = pd.to_datetime(df["timestamp_created"]).dt.date

        # Agregar control de selección de fecha
        start_date = st.date_input(f"Seleccione la fecha de inicio. Fecha minima de la Base de Datos: {min(df['timestamp_created'])}",value=min(df['timestamp_created']),min_value= min(df['timestamp_created']),max_value= max(df['timestamp_created']))
        end_date = st.date_input(f"Seleccione la fecha de fin. Fecha maxima de la Base de Datos: {max(df['timestamp_created'])}",value=max(df['timestamp_created']), min_value=min(df['timestamp_created']),max_value= max(df['timestamp_created']))
        
        # Filtrar DataFrame por rango de fechas
        df = df[(df['timestamp_created'] >= start_date) & (df['timestamp_created'] <= end_date)]

        # Botón de alternancia
        view_option = st.radio("**Seleccione el dashboard:**", ("Estadistica", "Modelos de mineria"))

        if view_option == "Estadistica":
            st.subheader("Estadistica")
            groupby_column = st.selectbox(
            '**¿Qué estadística te gustaría obtener?**',
            ('language','recommended','steam_purchase','received_for_free','timestamp_updated')
        )
            df_grouped = df[groupby_column].value_counts()

            # - - - - - -- - - -- Language - -- - - - - -- - - 
            # Gráfico de barras
            fig_bar = px.bar(
                df_grouped,
                color=df_grouped,
                color_continuous_scale=['grey','brown','black'],
                template='plotly_white',
                title=f'<b>Número de reseñas por {groupby_column}</b>'
            )
            # Actualizar el label del eje X
            fig_bar.update_xaxes(title_text=groupby_column)

            # Actualizar el label del eje Y
            fig_bar.update_yaxes(title_text="Cantidad de reseñas")

            # Configurar las etiquetas de datos
            fig_bar.update_traces(
                hovertemplate="Idioma: %{x}<br>Cantidad: %{y}",
            )

            # Mostrar el gráfico de barras
            st.plotly_chart(fig_bar)

            #Barras1
            if groupby_column == 'language' and 'language' in df.columns:
                language_distribution = df['language'].value_counts()
                st.markdown("**Análisis de Frecuencia de Reseñas por Idioma:**")
                st.markdown(f"**El idioma con más reseñas es:** {language_distribution.idxmax()} con {language_distribution.max()} reseñas")
                st.markdown(f"**El idioma con menos reseñas es:** {language_distribution.idxmin()} con {language_distribution.min()} reseñas")

            #Pie 2
            # Boxplot para métricas descriptivas adicionales por idioma
            if groupby_column == 'language' and 'language' in df.columns:
                # Agrupa por idioma y suma la cantidad de votos divertidos
                language_votes = df.groupby('language')['votes_funny'].sum()

                # Graficar la cantidad total de votos divertidos por cada idioma en un gráfico de pastel
                fig_pie_votes = px.pie(
                    names=language_votes.index,
                    values=language_votes.values,
                    title='Votos Divertidos por Language',
                    template='plotly_white'
                )
                st.plotly_chart(fig_pie_votes)

                # Obtener información sobre los votos divertidos
                max_lang_votes = language_votes.idxmax()
                min_lang_votes = language_votes.idxmin()

                # Texto informativo debajo del pie
                st.markdown("**Análisis de Votos Divertidos por Idioma:**")
                st.markdown(f"**El idioma con más votos divertidos es:** {max_lang_votes} con {language_votes.loc[max_lang_votes]} votos")
                st.markdown(f"**El idioma con menos votos divertidos es:** {min_lang_votes} con {language_votes.loc[min_lang_votes]} votos")

            # Distribucion invertidad 3
            # Gráfico de barra horizontal para juegos gratis recibidos por idioma
            if groupby_column == 'language' and 'language' in df.columns:
                # Filtra solo las filas donde el juego fue recibido gratis
                juegos_gratis = df[df['received_for_free'] == True]

                # Agrupa por idioma y cuenta la cantidad de juegos gratis recibidos
                language_counts = juegos_gratis['language'].value_counts()

                # Graficar la cantidad de juegos gratis recibidos por cada idioma en un gráfico de barra horizontal
                fig_bar = px.bar(
                    x=language_counts.values,
                    y=language_counts.index,
                    orientation='h',
                    title='Cantidad de Juegos Gratis Recibidos por language',
                    labels={'x': 'Cantidad de Juegos Gratis', 'y': 'Idioma'},
                    template='plotly_white'
                )
                st.plotly_chart(fig_bar)

                # Obtener información sobre los juegos gratis
                max_lang = language_counts.idxmax()
                min_lang = language_counts.idxmin()

                # Texto informativo debajo del gráfico de barra horizontal
                st.markdown("**Análisis de Juegos Gratis Recibidos por Idioma:**")
                st.markdown(f"**El idioma con más juegos gratis es:** {max_lang} con {language_counts[max_lang]} juegos recibidos")
                st.markdown(f"**El idioma con menos juegos gratis es:** {min_lang} con {language_counts[min_lang]} juegos recibidos")

            # - -- - -- -- -  -- - Columna de Recommended - - - - - - -  - --  - -- - --  - -- - 
            
            # Barras1
            if groupby_column == 'recommended' and 'recommended' in df.columns:
                language_distribution = df['recommended'].value_counts()
                st.markdown("**Análisis de Frecuencia de Reseñas por Recomendacion:**")
                st.markdown(f"**Las reseñas recomendadas fueron:**  {language_distribution.max()} reseñas")
                st.markdown(f"**Las reseñas no recomendadas fueron:** {language_distribution.min()} reseñas")

            #Pie 2
            # Boxplot para métricas descriptivas adicionales por idioma
            if groupby_column == 'recommended' and 'recommended' in df.columns: 

                #Filtrar el DataFrame para obtener solo las filas con recomendaciones
                filtered_df = df[df['recommended'].notnull()]

                # Contar la cantidad de recomendaciones positivas por idioma
                positive_recommendations_count = filtered_df[filtered_df['recommended'] == True].groupby('language').size().reset_index(name='count')

                # Graficar la cantidad de recomendaciones positivas en un gráfico de pastel
                fig_pie = px.pie(
                    positive_recommendations_count,
                    names='language',
                    values='count',
                    title='Frecuencia de mas Recomendaciones por Idioma',
                    template='plotly_white',
                )

                # Mostrar el gráfico de pastel
                st.plotly_chart(fig_pie)

                # Obtener información sobre el idioma con más recomendaciones positivas
                most_frequent_language_df = positive_recommendations_count.nlargest(1, 'count')

                if not most_frequent_language_df.empty:
                    most_frequent_language = most_frequent_language_df['language'].values[0]
                    # Texto informativo debajo del gráfico de pastel
                    st.markdown("**Análisis de Recomendaciones por Idioma:**")
                    st.markdown(f"**El que recommendo mas el juego fue el idioma:** {most_frequent_language}")

            #Grafica 3
            # Barras agrupadas para mostrar recomendaciones por compra del juego
            if groupby_column =='recommended' in df.columns and 'recommended' in df.columns:
                # Crear un DataFrame para el análisis
                df_analysis = df[['recommended', 'steam_purchase']]

                # Contar la cantidad de recomendaciones para cada categoría
                recommendations_count = df_analysis.groupby(['steam_purchase', 'recommended']).size().reset_index(name='count')

                # Filtrar por recomendaciones positivas y negativas cuando el juego fue comprado
                positive_recommendations = recommendations_count[(recommendations_count['recommended'] == 1) & (recommendations_count['steam_purchase'] == 1)]
                negative_recommendations = recommendations_count[(recommendations_count['recommended'] == 0) & (recommendations_count['steam_purchase'] == 1)]

                # Graficar la cantidad de recomendaciones en un gráfico de barras agrupadas
                fig_grouped_bar = go.Figure()

                # Agregar barras para recomendaciones positivas y negativas
                fig_grouped_bar.add_trace(go.Bar(x=positive_recommendations['steam_purchase'], y=positive_recommendations['count'], name='Recomendación Positiva', marker_color='green'))
                fig_grouped_bar.add_trace(go.Bar(x=negative_recommendations['steam_purchase'], y=negative_recommendations['count'], name='Recomendación Negativa', marker_color='red'))

                # Configurar diseño del gráfico
                fig_grouped_bar.update_layout(
                    barmode='group',
                    template='plotly_white',
                    title='Recomendaciones en función de la Compra del juego',
                    xaxis_title='Juego Comprado',
                    yaxis_title='Recomendaciones',
                )

                # Mostrar el gráfico de barras agrupadas
                st.plotly_chart(fig_grouped_bar)

                # Obtener información sobre las recomendaciones y compras
                max_compra_helpful = positive_recommendations['count'].max()
                max_no_compra_helpful = negative_recommendations['count'].max()

                # Texto informativo debajo del gráfico de barras agrupadas
                st.markdown("**Análisis de Recomendaciones por Compra del juego:**")
                st.markdown(f"**Recomendaciones al comprar el juego:** {max_compra_helpful}")
                st.markdown(f"**Sin Recomendaciones al comprar el juego:** {max_no_compra_helpful}")
                        

            # - -- - -- -- -  -- - Columna de steam_purcheased - - - - - - -  - --  - -- - --  - -- - 
            # Gráfico de barras 1
            if groupby_column == 'steam_purchase' and 'steam_purchase' in df.columns:
                language_distribution = df['steam_purchase'].value_counts()
                st.markdown("**Análisis de Frecuencia de Reseñas por Compras del juego:**")
                st.markdown(f"**El total de reseñas al comprar el juego:**  {language_distribution.max()} reseñas")
                st.markdown(f"**El total de reseñas al no comprar el juegon:** {language_distribution.min()} reseñas")
            
            #Grafica pastel 2
            if groupby_column == 'steam_purchase' and 'steam_purchase' in df.columns:
                # Calcular la proporción de compras y no compras
                purchase_percentage = df['steam_purchase'].value_counts(normalize=True) * 100

                # Obtener el total de usuarios
                total_users = len(df)

                # Crear un DataFrame para los datos del gráfico de pastel
                pie_data = pd.DataFrame({
                    'Categoría': ['Comprado', 'No Comprado'],
                    'Porcentaje': purchase_percentage.values
                })

                # Graficar pastel
                fig = px.pie(
                    pie_data,
                    names='Categoría',
                    values='Porcentaje',
                    title='Proporción de Compras en Steam',
                    template='plotly_white',
                    labels={'Porcentaje': 'Porcentaje'},
                    hover_data=['Porcentaje']
                )

                # Agregar información al centro de la gráfica
                fig.add_annotation(
                    text=f"Total de Usuarios: {total_users}",
                    showarrow=False
                )

                # Mostrar gráfico en Streamlit
                st.plotly_chart(fig)

                # Obtener información en texto
                comprado = df['steam_purchase'].sum()
                no_comprado = len(df) - comprado

                # Mostrar información en texto
                st.markdown("**Resumen de Compras en Steam:**")
                st.markdown(f"**Compraron el juego:** {comprado} usuarios ({purchase_percentage[True]:.2f}%)")
                st.markdown(f"**No compraron el juego:** {no_comprado} usuarios ({purchase_percentage[False]:.2f}%)")

            #Grafica plotty 3
            if groupby_column == 'steam_purchase' and 'steam_purchase' in df.columns:
                # Filtrar DataFrame para obtener solo aquellos que compraron el juego
                purchased_df = df[df['steam_purchase'] == True]

                # Crear el gráfico de dispersión
                fig_scatter = px.scatter(
                    purchased_df,
                    x='author.playtime_forever',
                    y=purchased_df.index,
                    color='steam_purchase',
                    title='Promedio de Horas jugadas de los que Compraron el juego',
                    template='plotly_white',
                    labels={'author.playtime_forever': 'Horas de Juego', 'index': 'Índice'}
                )

                # Configurar diseño del gráfico de dispersión
                fig_scatter.update_layout(
                    xaxis_title='Horas de Juego',
                    yaxis_title='Jugadores',
                )
                # Mostrar el gráfico de dispersión en Streamlit
                st.plotly_chart(fig_scatter)

                # Obtener el promedio de las horas jugadas de los que compraron el juego
                average_playtime = purchased_df['author.playtime_forever'].mean()

                # Mostrar el promedio en Streamlit
                st.markdown("**Resumen de Promedio de horas jugadas:**")
                st.markdown(f"**Promedio de Horas Jugadas de los que Compraron el Juego:** {average_playtime:.2f} horas")
                        
            # - -- - -- -- -  -- - Columna de received_for_free - - - - - - -  - --  - -- - --  - -- - 
            # Gráfico de barras 1
            if groupby_column == 'received_for_free' and 'received_for_free' in df.columns:
                language_distribution = df['received_for_free'].value_counts()
                st.markdown("**Análisis de Frecuencia de Reseñas al recibir el juego gratis:**")
                st.markdown(f"**El total de reseñas con el juego gratis:**  {language_distribution.min()} reseñas")
                st.markdown(f"**El total de reseñas sin el juego gratis:** {language_distribution.max()} reseñas")

            #Grafico de pie 2
            if groupby_column =='received_for_free' in df.columns and 'received_for_free' in df.columns:
                # Filtrar personas que no compraron ni recibieron gratis el juego
                not_purchased_not_received = df[(df['steam_purchase'] == False) & (df['received_for_free'] == False)]

                # Obtener el total de personas que no compraron ni recibieron gratis el juego
                total_not_purchased_not_received = len(not_purchased_not_received)

                # Graficar pastel
                fig = px.pie(
                    names=['Ni Recibido Gratis, Ni Comprado', 'Otro'],
                    values=[total_not_purchased_not_received, len(df) - total_not_purchased_not_received],
                    title='Personas que No recibieron gratis ni lo compraron',
                    template='plotly_white'
                )
                # Mostrar gráfico en Streamlit
                st.plotly_chart(fig)

                # Mostrar información en texto
                st.markdown("**Resumen de Personas que No recibieron gratis ni lo compraron:**")
                st.markdown(f"**Total:** {total_not_purchased_not_received} personas")
                st.markdown(f"Estos usuarios no cuentan con el videojuego")
            
            # Gráfico 3
            if groupby_column == 'received_for_free' and 'received_for_free' in df.columns:
                # Filtrar DataFrame para obtener solo aquellos que recibieron el juego gratis
                free_df = df[df['received_for_free'] == True]

                # Crear el gráfico de dispersión
                fig_scatter_free = px.scatter(
                    free_df,
                    x='author.playtime_forever',
                    title='Promedio de Horas jugadas de los que Recibieron Gratis el juego',
                    template='plotly_white',
                    labels={'author.playtime_forever': 'Horas de Juego', 'index': 'Índice'}
                )

                # Configurar diseño del gráfico de dispersión
                fig_scatter_free.update_layout(
                    xaxis_title='Horas de Juego',
                    yaxis_title='Recibido Gratis',
                )

                # Mostrar el gráfico de dispersión en Streamlit
                st.plotly_chart(fig_scatter_free)

                # Obtener el promedio de las horas jugadas de los que recibieron el juego gratis
                average_playtime_free = free_df['author.playtime_forever'].mean()

                # Mostrar el promedio en Streamlit
                st.markdown("**Resumen de Promedio de horas jugadas para los que recibieron Gratis el Juego:**")
                st.markdown(f"**Promedio de Horas Jugadas:** {average_playtime_free:.2f} horas")

            # - -- - -- -- -  -- - Columna de timestamp_updated - - - - - - -  - --  - -- - --  - -- -
            # Gráfico de barras 1
            if groupby_column == 'timestamp_updated' and 'timestamp_updated' in df.columns:
                # Obtener la distribución de fechas de actualización
                timestamp_distribution = df['timestamp_updated'].value_counts()

                # Obtener la fecha donde se actualizó más y menos la reseña
                most_updated_date = timestamp_distribution.idxmax()
                least_updated_date = timestamp_distribution.idxmin()

                # Obtener el total de reseñas para esas fechas
                total_most_updated_reviews = timestamp_distribution.max()
                total_least_updated_reviews = timestamp_distribution.min()

                st.markdown("**Análisis de Reseñas según su fecha de actualizacion:**")
                st.markdown(f"**Fecha donde se actualizó más la reseña:** {most_updated_date} con {total_most_updated_reviews} reseñas")
                st.markdown(f"**Fecha donde se actualizó menos la reseña:** {least_updated_date} con {total_least_updated_reviews} reseñas")

            #Grafico de linea 2
            if groupby_column == 'timestamp_updated' and 'timestamp_updated' in df.columns:
                # Crear un DataFrame para la distribución de fechas de actualización
                timestamp_distribution_df = df['timestamp_updated'].value_counts().reset_index()
                timestamp_distribution_df.columns = ['timestamp_updated', 'count']

                # Ordenar el DataFrame por fecha
                timestamp_distribution_df = timestamp_distribution_df.sort_values('timestamp_updated')

                # Crear un gráfico de línea
                fig_line_chart = px.line(timestamp_distribution_df, x='timestamp_updated', y='count', title='Fecha Maxima de Actualización de Reseñas')

                # Configurar diseño del gráfico
                fig_line_chart.update_layout(template='plotly_white', xaxis_title='Fecha Maxima de Actualización', yaxis_title='Cantidad de Reseñas')

                # Mostrar el gráfico de línea en Streamlit
                st.plotly_chart(fig_line_chart)

                # Obtener la fecha mínima y máxima de la columna timestamp_updated
                max_timestamp_updated = df['timestamp_updated'].max()

                # Mostrar la información en Streamlit
                st.markdown(f"**Fecha máxima de actualización de reseñas:** {max_timestamp_updated}")

        # - - - - - - -- - - - -- - - Modelos de Mineria - - --- - -- - -- - - - -- - - - --  -- 

        elif view_option == "Modelos de mineria":        
            # Seleccionar la columna para la regresión logística
            regression_column = st.selectbox("**Seleccione la columna para la regresión logística:**", ['author.playtime_forever','votes_helpful','author.playtime_last_two_weeks', 'author.num_reviews','steam_purchase','received_for_free','author.num_games_owned'])

            # Crear y entrenar el modelo de regresión logística
            X = df[[regression_column]]
            y = df['recommended']

            # Divide los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Normalizar las características
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Crear el modelo de regresión logística
            model = LogisticRegression(max_iter=1000)
            
            # Entrenar el modelo
            model.fit(X_train_scaled, y_train)

            # Realizar predicciones en el conjunto de prueba
            y_pred = model.predict(X_test_scaled)

            # Evaluar el rendimiento del modelo
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)

            # Obtener las probabilidades de predicción para la clase positiva
            y_prob = model.predict_proba(X_test_scaled)[:, 1]

            # Calcular la curva ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            # Calcular la curva de precisión-recall
            precision, recall, _ = precision_recall_curve(y_test, y_prob)

            # Reporte
            st.markdown("**Resultados del Modelo de Regresión Logística:**")
            st.markdown(f"Exactitud (Accuracy): {accuracy:.2f}")

            # Graficar la curva ROC
            st.markdown("**Curva ROC:**")
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            st.pyplot(plt)
            st.text(f"El AUC (Área bajo la curva ROC) es {roc_auc:.2f}.")

            # Mostrar la curva de precisión-recall
            st.markdown("**Curva Precision-Recall:**")
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision, color='blue', lw=2, label='Curva Precision-Recall')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall')
            st.pyplot(plt)

            ## Graficar la matriz de confusión
            st.markdown("**Matriz de Confusión:**")
            plt.figure(figsize=(8, 6))
            sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
            plt.xlabel('Predicciones')
            plt.ylabel('Valores Verdaderos')
            plt.title('Matriz de Confusión')
            st.pyplot(plt)

            # Número de predicciones realizadas
            total_predictions = len(y_test)
            correct_predictions = conf_matrix[0, 0] + conf_matrix[1, 1]
            st.text(f"El modelo realizó {total_predictions} predicciones, de las cuales {correct_predictions} fueron correctas.")

            # Obtener los coeficientes del modelo
            coefficients = model.coef_[0]

            # Obtener el nombre de las características
            feature_names = X.columns

            # Crear un DataFrame para facilitar la visualización
            coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

            # Ordenar el DataFrame por valor absoluto de los coeficientes
            coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
