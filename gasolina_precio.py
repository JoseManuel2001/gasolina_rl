import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Precio Gasolina - Regresión Lineal", page_icon="⛽", layout="centered")

st.title("⛽ Predicción del precio de gasolina con Regresión Lineal (sklearn)")
st.caption("Archivo usado: **gasolina_precios (1).csv** | Variables: estado (categórica), año, mes → precio")

DATA_PATH = "gasolina_precios.csv"

@st.cache_data
def load_data():
    tries = [
        dict(sep=",", encoding="utf-8"),
        dict(sep=";", encoding="utf-8"),
        dict(sep=",", encoding="latin-1"),
    ]
    last_err = None
    df = None
    for kw in tries:
        try:
            tmp = pd.read_csv(DATA_PATH, **kw)
            tmp.columns = [c.strip().lower() for c in tmp.columns]
            if "anio" in tmp.columns and "año" not in tmp.columns:
                tmp = tmp.rename(columns={"anio": "año"})
            if {"estado", "año", "mes", "precio"}.issubset(tmp.columns):
                df = tmp[["estado", "año", "mes", "precio"]].copy()
                break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"No se pudo leer el CSV. Último error: {last_err}")
    df["estado"] = df["estado"].astype(str)
    df["año"] = pd.to_numeric(df["año"], errors="coerce")
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce")
    df["precio"] = pd.to_numeric(df["precio"], errors="coerce")
    df = df.dropna(subset=["estado", "año", "mes", "precio"]).reset_index(drop=True)
    return df

df = load_data()

with st.expander("👁️ Ver muestra del dataset", expanded=False):
    st.dataframe(df.head(20))

X = df[["estado", "año", "mes"]]
y = df["precio"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["estado"])
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("prep", preprocess),
    ("reg", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Métricas del modelo (test)")
col1, col2 = st.columns(2)
with col1:
    st.metric("RMSE", f"{rmse:,.4f}")
with col2:
    st.metric("R²", f"{r2:,.4f}")

st.header("🔮 Predicción para un nuevo caso")
estados = sorted(df["estado"].unique().tolist())

colA, colB = st.columns(2)
with colA:
    estado_in = st.selectbox("Estado", estados, index=0)
with colB:
    anio_in = st.number_input(
        "Año",
        min_value=int(df["año"].min()),
        max_value=int(df["año"].max()),
        value=int(df["año"].max()),
        step=1
    )

mes_in = st.number_input("Mes", min_value=1, max_value=12, value=int(df["mes"].median()), step=1)

input_df = pd.DataFrame([{
    "estado": estado_in,
    "año": int(anio_in),
    "mes": int(mes_in)
}])

if st.button("Predecir precio"):
    pred = model.predict(input_df)[0]
    st.success(f"Precio estimado: **{pred:,.4f}**")

with st.expander("🧠 Ver coeficientes (interpretación)", expanded=False):
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    try:
        ohe_feature_names = ohe.get_feature_names_out(["estado"]).tolist()
    except Exception:
        ohe_feature_names = list(ohe.get_feature_names(["estado"]))
    feature_names = ohe_feature_names + ["año", "mes"]
    coefs = model.named_steps["reg"].coef_
    intercept = model.named_steps["reg"].intercept_
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coefs}).sort_values("feature")
    st.write("Intercepto (β0):", round(float(intercept), 6))
    st.dataframe(coef_df.reset_index(drop=True))

