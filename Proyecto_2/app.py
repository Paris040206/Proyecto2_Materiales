"""
Explorador de aceros al carbono - Proyecto 2 (Ingeniería de materiales)

Cómo correrlo:
    pip install streamlit pandas plotly scipy statsmodels
    streamlit run app.py

El CSV debe estar en la misma carpeta que este archivo.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from pathlib import Path

ARCHIVO = next(Path(__file__).parent.glob("*.csv"))

PROPIEDADES = {
    "UTS": "Resistencia última a la tensión, UTS (MPa)",
    "YS": "Esfuerzo de fluencia, YS (MPa)",
    "Dureza": "Dureza (HB)",
    "Elongacion": "Elongación (%)",
}


# ============================================================
# 1. CARGA Y LIMPIEZA (versión provisional: la refinamos después)
# ============================================================

# --- INICIO LIMPIEZA ---
def clasificar_condicion(texto):
    """Agrupa las 46 etiquetas originales en pocas categorías."""
    t = str(texto).lower()
    estirado = "cold" in t or "orawn" in t  # "Cold Orawn" es un error de captura

    if "quench" in t or "temper" in t:
        return "Templado y revenido"
    if "normal" in t:
        return "Normalizado + estirado en frío" if estirado else "Normalizado"
    if "anneal" in t:
        return "Recocido + estirado en frío" if estirado else "Recocido"
    if estirado:
        return "Estirado en frío"
    if "hot" in t or "rot" in t:  # "Hot rotted" es un error de captura
        return "Laminado en caliente"
    return "Otro"


def limpiar(df):
    # Nombres de columnas más cómodos
    df = df.rename(columns={
        "SAE Grade": "SAE",
        "Conditions": "Condicion_original",
        "UTS (MPa)": "UTS",
        "YS (MPa)": "YS",
        "Elongation (%)": "Elongacion",
        "Reduction (%)": "Reduccion",
        "Hardness (HB)": "Dureza",
        "C (Min)": "C_min",
        "C (Max)": "C_max",
    })
    df = df[["SAE", "Condicion_original", "UTS", "YS", "Elongacion",
             "Reduccion", "Dureza", "C_min", "C_max"]].copy()

    # Reducción: algunos valores usan coma decimal ("40,2")
    df["Reduccion"] = pd.to_numeric(
        df["Reduccion"].astype(str).str.replace(",", "."), errors="coerce")

    # Dureza: dos registros vienen en Rockwell B ("269 HRB"), no en Brinell.
    # Con to_numeric quedan como NaN para no mezclar escalas.
    df["Dureza"] = pd.to_numeric(df["Dureza"], errors="coerce")

    # C_max = 15 en el acero 12L14 es un punto decimal perdido (0.15 %)
    df.loc[df["C_max"] > 2, "C_max"] = df["C_max"] / 100

    # %C = punto medio del rango (si falta el mínimo, queda solo el máximo)
    df["C_prom"] = df[["C_min", "C_max"]].mean(axis=1)

    df["Condicion"] = df["Condicion_original"].apply(clasificar_condicion)
    return df
# --- FIN LIMPIEZA ---


@st.cache_data
def cargar_datos():
    return limpiar(pd.read_csv(ARCHIVO))


# ============================================================
# 2. CONFIGURACIÓN DE LA PÁGINA Y BARRA LATERAL
# ============================================================

st.set_page_config(page_title="Aceros al carbono", page_icon="🔩", layout="wide")
st.title("🔩 Explorador de aceros al carbono")
st.caption("Composición química, procesamiento y propiedades mecánicas")

df = cargar_datos()

st.sidebar.header("Filtros")
condiciones = st.sidebar.multiselect(
    "Condición de procesamiento",
    options=sorted(df["Condicion"].unique()),
    default=sorted(df["Condicion"].unique()),
)
c_min_global = float(df["C_prom"].min())
c_max_global = float(df["C_prom"].max())
rango_c = st.sidebar.slider(
    "Rango de %C", c_min_global, c_max_global, (c_min_global, c_max_global))

dff = df[df["Condicion"].isin(condiciones)
         & df["C_prom"].between(rango_c[0], rango_c[1])]
st.sidebar.write(f"Registros mostrados: **{len(dff)}** de {len(df)}")

if len(dff) == 0:
    st.warning("Los filtros no dejan ningún registro. Amplía la selección.")
    st.stop()

tabs = st.tabs([
    "📋 Datos", "📈 %C vs propiedades", "⚙️ Procesamiento",
    "🔥 Recocido vs normalizado", "📊 ANOVA", "🔎 Buscador de acero",
])

# ============================================================
# 3. PESTAÑA: DATOS
# ============================================================
with tabs[0]:
    st.subheader("Datos y calidad de la información")
    col1, col2, col3 = st.columns(3)
    col1.metric("Registros", len(df))
    col2.metric("Grados SAE distintos", df["SAE"].nunique())
    col3.metric("Etiquetas originales de condición", df["Condicion_original"].nunique())

    st.markdown("""
**¿Qué representa cada variable?**
- **%C:** contenido de carbono; controla la cantidad de perlita (y por tanto la resistencia).
- **UTS:** esfuerzo máximo que soporta antes de romper.
- **YS:** esfuerzo a partir del cual hay deformación plástica.
- **Dureza (HB):** resistencia a la indentación (escala Brinell).
- **Elongación (%):** medida de ductilidad.
    """)

    with st.expander("Decisiones de limpieza (completar después)"):
        st.markdown("""
- Las 46 etiquetas de condición se agruparon en pocas categorías y se corrigieron errores de captura.
- Valores con coma decimal se convirtieron a punto.
- Dureza en escala HRB se excluyó para no mezclar escalas.
- El %C se tomó como el punto medio del rango mínimo-máximo.
        """)

    st.dataframe(dff, use_container_width=True)
    st.write("Registros por condición:")
    st.bar_chart(df["Condicion"].value_counts())

# ============================================================
# 4. PESTAÑA: %C VS PROPIEDADES
# ============================================================
with tabs[1]:
    st.subheader("Efecto del contenido de carbono")
    prop = st.selectbox("Propiedad", list(PROPIEDADES), format_func=PROPIEDADES.get)

    datos = dff.dropna(subset=["C_prom", prop])
    fig = px.scatter(
        datos, x="C_prom", y=prop, color="Condicion",
        hover_data=["SAE", "Condicion_original"],
        trendline="ols", trendline_scope="overall",
        labels={"C_prom": "%C (punto medio)", prop: PROPIEDADES[prop]},
    )
    st.plotly_chart(fig, use_container_width=True)

    if len(datos) > 2:
        r, p = stats.pearsonr(datos["C_prom"], datos[prop])
        st.write(f"Correlación de Pearson: **r = {r:.2f}** (valor p = {p:.4f})")
    st.info("💡 Interpretación (completar): más carbono → más perlita → "
            "mayor UTS y dureza, pero menor elongación.")

# ============================================================
# 5. PESTAÑA: PROCESAMIENTO
# ============================================================
with tabs[2]:
    st.subheader("Comparación entre condiciones de procesamiento")
    prop2 = st.selectbox("Propiedad ", list(PROPIEDADES),
                         format_func=PROPIEDADES.get, key="prop_proc")
    fig = px.box(dff, x="Condicion", y=prop2, color="Condicion", points="all",
                 hover_data=["SAE"], labels={prop2: PROPIEDADES[prop2]})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    resumen = dff.groupby("Condicion")[prop2].agg(["count", "mean", "std"]).round(1)
    resumen.columns = ["n", "media", "desv. estándar"]
    st.dataframe(resumen, use_container_width=True)
    st.info("💡 Interpretación (completar): el estirado en frío endurece por "
            "deformación (sube YS y UTS, baja elongación); el recocido ablanda; "
            "el normalizado refina el grano.")

# ============================================================
# 6. PESTAÑA: RECOCIDO VS NORMALIZADO
# ============================================================
with tabs[3]:
    st.subheader("Recocido vs normalizado")
    sub = df[df["Condicion"].isin(["Recocido", "Normalizado"])]
    prop3 = st.selectbox("Propiedad  ", list(PROPIEDADES),
                         format_func=PROPIEDADES.get, key="prop_rn")
    fig = px.scatter(sub, x="C_prom", y=prop3, color="Condicion",
                     hover_data=["SAE", "Condicion_original"],
                     labels={"C_prom": "%C (punto medio)", prop3: PROPIEDADES[prop3]})
    st.plotly_chart(fig, use_container_width=True)
    st.write(sub.groupby("Condicion")[prop3].agg(["count", "mean"]).round(1))
    st.warning("⚠️ Limitaciones (completar): pocos registros por grupo, aceros "
               "distintos en cada grupo y temperaturas de tratamiento diferentes.")

# ============================================================
# 7. PESTAÑA: ANOVA
# ============================================================
with tabs[4]:
    st.subheader("ANOVA de un factor: ¿la condición afecta la propiedad?")
    prop4 = st.selectbox("Propiedad   ", list(PROPIEDADES),
                         format_func=PROPIEDADES.get, key="prop_anova")

    # Solo grupos con al menos 3 datos
    grupos = {nombre: g[prop4].dropna().values
              for nombre, g in df.groupby("Condicion")
              if g[prop4].notna().sum() >= 3}

    if len(grupos) < 2:
        st.error("No hay suficientes grupos con datos para hacer ANOVA.")
    else:
        f, p = stats.f_oneway(*grupos.values())
        h, p_kw = stats.kruskal(*grupos.values())
        _, p_lev = stats.levene(*grupos.values())

        c1, c2, c3 = st.columns(3)
        c1.metric("ANOVA: F", f"{f:.2f}")
        c2.metric("ANOVA: valor p", f"{p:.4g}")
        c3.metric("Levene (varianzas): p", f"{p_lev:.4g}")
        st.write(f"Prueba alternativa Kruskal-Wallis: valor p = {p_kw:.4g}")

        if p < 0.05:
            st.success("Hay diferencias significativas entre al menos dos condiciones (p < 0.05).")
        else:
            st.info("No se detectan diferencias significativas (p ≥ 0.05).")
        if p_lev < 0.05:
            st.warning("Las varianzas no son homogéneas: el ANOVA clásico es cuestionable.")
        st.caption("Ojo: los datos son valores nominales de manual, no réplicas "
                   "experimentales; la independencia de las observaciones es discutible.")

# ============================================================
# 8. PESTAÑA: BUSCADOR DE ACERO (propuesta propia)
# ============================================================
with tabs[5]:
    st.subheader("¿Qué acero cumple mis requisitos?")
    c1, c2, c3 = st.columns(3)
    uts_min = c1.number_input("UTS mínimo (MPa)", min_value=0, value=500, step=10)
    ys_min = c2.number_input("YS mínimo (MPa)", min_value=0, value=300, step=10)
    elong_min = c3.number_input("Elongación mínima (%)", min_value=0.0, value=15.0, step=1.0)

    ok = df[(df["UTS"] >= uts_min) & (df["YS"] >= ys_min) & (df["Elongacion"] >= elong_min)]
    ok = ok.sort_values("UTS", ascending=False)
    st.write(f"**{len(ok)}** registros cumplen los requisitos:")
    st.dataframe(ok[["SAE", "Condicion", "C_prom", "UTS", "YS", "Elongacion", "Dureza"]],
                 use_container_width=True)

    # Comparador de dos aceros (gráfico de radar normalizado)
    st.markdown("---")
    st.subheader("Comparador de dos registros")
    df["Etiqueta"] = df["SAE"].astype(str) + " | " + df["Condicion"]
    a = st.selectbox("Acero A", df["Etiqueta"].unique(), index=0)
    b = st.selectbox("Acero B", df["Etiqueta"].unique(), index=min(60, len(df) - 1))

    props = ["UTS", "YS", "Dureza", "Elongacion"]
    maximos = df[props].max()

    def valores(etiqueta):
        fila = df[df["Etiqueta"] == etiqueta].iloc[0]
        return [fila[p] / maximos[p] for p in props]

    radar = go.Figure()
    for etiqueta in (a, b):
        v = valores(etiqueta)
        radar.add_trace(go.Scatterpolar(r=v + [v[0]], theta=props + [props[0]],
                                        fill="toself", name=etiqueta))
    radar.update_layout(polar=dict(radialaxis=dict(range=[0, 1])))
    st.plotly_chart(radar, use_container_width=True)
    st.caption("Cada propiedad está dividida entre su máximo en la base de datos (escala 0 a 1).")
