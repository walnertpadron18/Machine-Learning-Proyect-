import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor)
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------------
# Configuración y constantes
# ------------------------------------------------------------------
st.set_page_config(page_title="Alquiler de bicicletas", layout="wide")

CARPETA_DATOS = Path(__file__).parent / "Data"
COLUMNAS = ['yr', 'mnth', 'hr', 'workingday', 'weathersit', 'temp', 'windspeed']
COLUMNAS_FUGA = ['casual', 'registered', 'instant']

CLIMA = {
    1: "1 - Despejado",
    2: "2 - Nublado",
    3: "3 - Lluvia o nieve ligera",
    4: "4 - Lluvia o nieve fuerte",
}


# ------------------------------------------------------------------
# Carga de datos
# ------------------------------------------------------------------
@st.cache_data
def cargar_datos():
    return pd.read_csv(CARPETA_DATOS / "hour.csv")


@st.cache_data
def cargar_resultados_ajustados():
    """Lee los resultados de Grid y Random Search guardados desde el notebook."""
    ruta = CARPETA_DATOS / "resultados_modelos.csv"
    if not ruta.exists():
        return None
    return pd.read_csv(ruta)


df = cargar_datos()

faltan = [c for c in COLUMNAS + ['cnt'] if c not in df.columns]
if faltan:
    st.error(f"Faltan estas columnas en el archivo: {faltan}")
    st.stop()


# ------------------------------------------------------------------
# Modelos base (se entrenan una sola vez)
# ------------------------------------------------------------------
@st.cache_resource
def entrenar_modelos_base():
    X = pd.get_dummies(df[COLUMNAS], columns=['weathersit'])
    y = df['cnt']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelos = {
        "Regresión lineal": LinearRegression(),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5)),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Bagging": BaggingRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=100, max_samples = 1000),
        "AdaBoost": AdaBoostRegressor(DecisionTreeRegressor(max_depth=20), n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(max_depth=20, n_estimators=100)  
    }

    resultados = {}
    for nombre, m in modelos.items():
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
        resultados[nombre] = {
            "modelo": m,
            "pred": pred,
            "R² train": m.score(X_train, y_train),
            "R² test": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "% error ≤ 50": (np.abs(pred - y_test.values) <= 50).mean(),
        }
    return resultados, y_test, list(X.columns)


resultados, y_test, columnas_X = entrenar_modelos_base()


# ------------------------------------------------------------------
# Cabecera
# ------------------------------------------------------------------
st.title("Predicción de alquiler de bicicletas")
st.caption(
    "UCI Bike Sharing Dataset (hour.csv): 17.379 horas de 2011 y 2012. "
    "Objetivo: predecir cuántas bicis se alquilan cada hora (cnt)."
)

tab_expl, tab_base, tab_ajust, tab_pred = st.tabs([
    "Exploración",
    "Modelos base",
    "Modelos ajustados",
    "Predicción",
])


# ------------------------------------------------------------------
# 1. Exploración
# ------------------------------------------------------------------
with tab_expl:
    st.subheader("Mapa de correlaciones")
    ocultar = st.checkbox("Ocultar casual, registered e instant", value=False)

    datos = df.drop(columns=COLUMNAS_FUGA, errors="ignore") if ocultar else df
    corr = datos.select_dtypes(include="number").corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, mask=mask, cmap=sns.diverging_palette(220, 10, as_cmap=True),
        vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
        cbar_kws={"shrink": .5}, annot=True, fmt=".2f", ax=ax,
    )
    st.pyplot(fig)
    st.caption(
        "casual y registered suman cnt, por eso no entran en el modelo (fuga de datos). "
        "temp y atemp tienen 0,99 de correlación, así que solo se usa temp."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Demanda media por hora")
        st.line_chart(df.groupby("hr")["cnt"].mean())
    with col_b:
        st.subheader("Demanda según la temperatura")
        temp_c = (df["temp"] * 47 - 8).round()
        st.line_chart(df.groupby(temp_c)["cnt"].mean())
        st.caption("Temperatura aproximada en °C (la columna viene normalizada).")


# ------------------------------------------------------------------
# 2. Modelos base
# ------------------------------------------------------------------
with tab_base:
    st.subheader("Comparación de modelos base")
    st.caption("Mismo split (80/20, random_state=42) y mismas columnas para los tres.")

    tabla_base = pd.DataFrame({
        n: {k: v for k, v in r.items() if k not in ("modelo", "pred")}
        for n, r in resultados.items()
    }).T
    st.dataframe(
        tabla_base.style.format({
            "R² train": "{:.2f}", "R² test": "{:.2f}",
            "MAE": "{:.0f}", "RMSE": "{:.0f}", "% error ≤ 50": "{:.0%}",
        }),
        use_container_width=True,
    )
    st.bar_chart(tabla_base["R² test"])

    st.divider()
    st.subheader("Detalle de un modelo")
    elegido = st.selectbox("Elige un modelo", list(resultados), key="detalle_base")
    r = resultados[elegido]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (test)", f"{r['R² test']:.2f}")
    c2.metric("MAE", f"{r['MAE']:.0f} bicis")
    c3.metric("RMSE", f"{r['RMSE']:.0f} bicis")
    c4.metric("Error de 50 bicis o menos", f"{r['% error ≤ 50']:.0%}")

    st.scatter_chart(
        pd.DataFrame({"Real": y_test.values, "Predicho": r["pred"]}),
        x="Real", y="Predicho",
    )

    if elegido == "Random Forest":
        st.subheader("Qué variables pesan más")
        imp = pd.Series(
            r["modelo"].feature_importances_, index=columnas_X
        ).sort_values(ascending=False)
        st.bar_chart(imp)


# ------------------------------------------------------------------
# 3. Modelos ajustados (Grid Search y Random Search)
# ------------------------------------------------------------------
with tab_ajust:
    st.subheader("Modelos ajustados con Grid Search y Random Search")

    res = cargar_resultados_ajustados()

    if res is None:
        st.info(
            "Todavía no hay resultados guardados. Ejecuta las búsquedas en el notebook "
            "y guarda la tabla en Data/resultados_modelos.csv (columnas: Modelo, Búsqueda, "
            "R2 CV, R2 test, MAE test, Mejores parámetros)."
        )
    else:
        tipos = sorted(res["Búsqueda"].unique())
        elegidos = st.multiselect("Tipo de búsqueda", tipos, default=tipos)
        filtrado = res[res["Búsqueda"].isin(elegidos)]

        if filtrado.empty:
            st.warning("Elige al menos un tipo de búsqueda.")
        else:
            mejor = filtrado.sort_values("R2 test", ascending=False).iloc[0]
            base_rf = resultados["Random Forest"]["R² test"]

            c1, c2, c3 = st.columns(3)
            c1.metric("Mejor modelo", f"{mejor['Modelo']} ({mejor['Búsqueda']})")
            c2.metric("R² (test)", f"{mejor['R2 test']:.3f}",
                      delta=f"{mejor['R2 test'] - base_rf:+.3f} frente a Random Forest sin ajustar")
            c3.metric("MAE", f"{mejor['MAE test']:.0f} bicis")

            st.dataframe(
                filtrado.drop(columns=["Mejores parámetros"])
                        .sort_values("R2 test", ascending=False),
                hide_index=True, use_container_width=True,
            )

            st.bar_chart(
                filtrado.pivot_table(index="Modelo", columns="Búsqueda", values="R2 test")
            )

            with st.expander("Mejores parámetros de cada modelo"):
                for _, fila in filtrado.sort_values("R2 test", ascending=False).iterrows():
                    st.markdown(f"**{fila['Modelo']}** ({fila['Búsqueda']})")
                    st.code(str(fila["Mejores parámetros"]))

            st.caption(
                "R2 CV es la media de la validación cruzada sobre train; R2 test se calcula "
                "una sola vez con datos que el modelo no ha visto."
            )


# ------------------------------------------------------------------
# 4. Predicción interactiva
# ------------------------------------------------------------------
with tab_pred:
    st.subheader("Predice los alquileres de una hora")

    modelo_pred = st.selectbox(
        "Modelo", list(resultados), index=list(resultados).index("Random Forest"),
        key="modelo_pred",
    )

    c1, c2 = st.columns(2)
    with c1:
        hr = st.slider("Hora", 0, 23, 8)
        mes = st.slider("Mes", 1, 12, 6)
        anio = st.selectbox("Año", [2011, 2012], index=1)
        laborable = st.selectbox(
            "Día laborable", [1, 0], format_func=lambda v: "Sí" if v else "No"
        )
    with c2:
        clima = st.selectbox("Tiempo", list(CLIMA), format_func=lambda v: CLIMA[v])
        temp_c = st.slider("Temperatura aproximada (°C)", -8, 39, 20)
        viento = st.slider("Viento aproximado (km/h)", 0, 67, 12)

    entrada = pd.DataFrame([{
        "yr": anio - 2011,
        "mnth": mes,
        "hr": hr,
        "workingday": laborable,
        "weathersit": clima,
        "temp": (temp_c + 8) / 47,
        "windspeed": viento / 67,
    }])[COLUMNAS]

    entrada = pd.get_dummies(entrada, columns=["weathersit"]).reindex(
        columns=columnas_X, fill_value=0
    )

    pred = resultados[modelo_pred]["modelo"].predict(entrada)[0]
    st.metric("Bicis esperadas en esa hora", f"{max(pred, 0):.0f}")
    st.caption(
        "Temperatura y viento se convierten a la escala normalizada del dataset. "
        "La regresión lineal puede dar valores negativos; aquí se muestran como 0."
    )