# Predicción inteligente de la demanda de bicicletas.

## Objetivo
Predecir cuántas bicicletas se alquilarán cada hora a partir del calendario y del clima, para anticipar la demanda y repartir mejor la flota.

## Contexto del negocio
Empresa que opera un sistema de bicicletas compartidas en una ciudad. Su reto es tener bicis disponibles en las horas punta sin dejar flota parada en las horas bajas. Decisiones que apoya el modelo: cuántas bicis reponer y cuándo.

## Dataset
- **Fuente:** UCI Bike Sharing Dataset (`hour.csv`), datos reales de 2011 y 2012, 17.379 filas (una por hora).
- **Variable objetivo:** `cnt` (bicis alquiladas por hora).
- **Variables usadas:** `yr`, `mnth`, `hr`, `workingday`, `weathersit`, `temp`, `windspeed`.
- **Excluidas:** `casual` y `registered` (suman `cnt`: fuga de datos), `instant` (contador de filas), `atemp` (correlación 0,99 con `temp`).
- **Calidad del dato:** sin nulos (comprobado con `df.isna().sum()`). `temp`, `hum` y `windspeed` vienen normalizadas.

## Preguntas clave
1. ¿Qué factores explican la demanda horaria?
2. ¿Qué modelo la predice mejor?
3. ¿Cuánto se equivoca en las horas punta?

## Proceso de análisis
EDA y mapa de correlaciones → selección de columnas → dummies en `weathersit` → split train/test (80/20) → modelos: regresión lineal, KNN y Random Forest (ajustados con Grid Search y Random Search) → métricas: R², MAE y RMSE.

## Resultados

- La hora del día y la temperatura son las variables que más pesan.
- La demanda tiene dos picos (mañana y tarde) que una recta no puede reproducir, y por eso la lineal queda muy por debajo.
- En 2012 hubo más alquileres que en 2011.
- Random Forest acierta con un error de 50 bicis o menos en el 85 % de las horas.

## Recomendaciones de negocio
- Reforzar la reposición de bicis antes de las horas punta.
- Reducir flota disponible con mal tiempo o frío.
- Priorizar los días laborables a primera y última hora.

## Limitaciones
- El split aleatorio es algo optimista en datos temporales.
- El modelo infravalora las horas de demanda muy alta.
- Solo dos años de datos y sin información de eventos ni de ubicación de estaciones.

## Próximos pasos
Validar con un split temporal (entrenar en 2011, probar en 2012), probar Gradient Boosting, incorporar festivos y eventos, y publicar la app de Streamlit.

## Cómo replicar el proyecto
- Notebook: `Machine1.ipynb`
- Limpieza: `cleaning.py`
- App: `python -m streamlit run app.py`