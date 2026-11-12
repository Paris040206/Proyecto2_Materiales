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

opciones = sorted(df["Condicion"].unique())
conteo = df["Condicion"].value_counts()

condiciones = st.sidebar.pills(
    "Condición de procesamiento",
    options=opciones,
    selection_mode="multi",
    default=opciones,
    format_func=lambda c: f"{c} ({conteo[c]})",
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

    with st.expander("❓ ¿Qué es el ANOVA? (explicación sencilla)"):
        st.markdown("""
El ANOVA responde una pregunta: **¿las diferencias entre los promedios de varios grupos
son reales, o podrían ser pura casualidad?**

Aquí los grupos son las condiciones de procesamiento (recocido, estirado en frío, etc.)
y comparamos el promedio de la propiedad elegida.

**Cómo leer el resultado:**
1. Mira el **valor p**. Si es **menor a 0.05**, hay diferencias reales entre grupos.
   Si es **0.05 o mayor**, no hay evidencia suficiente para afirmarlo.
2. Revisa si las pruebas de **Levene** y **Kruskal-Wallis** apoyan la conclusión.
3. Lee la sección "En palabras sencillas" y las advertencias antes de concluir.
        """)

    prop4 = st.selectbox("Propiedad   ", list(PROPIEDADES),
                         format_func=PROPIEDADES.get, key="prop_anova")

    nombres = {"UTS": "la resistencia última (UTS)", "YS": "el esfuerzo de fluencia (YS)",
               "Dureza": "la dureza", "Elongacion": "la elongación"}
    unidades = {"UTS": "MPa", "YS": "MPa", "Dureza": "HB", "Elongacion": "%"}

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

        # ---------- Resultado formal ----------
        st.markdown("#### 📐 Resultado formal")
        c1, c2, c3 = st.columns(3)
        c1.metric("ANOVA: F", f"{f:.2f}")
        c2.metric("ANOVA: valor p", f"{p:.4g}")
        c3.metric("Levene (varianzas): p", f"{p_lev:.4g}")
        st.write(f"Prueba alternativa Kruskal-Wallis: valor p = {p_kw:.4g}")

        # ---------- Resultado en palabras sencillas ----------
        st.markdown("#### 🗣️ En palabras sencillas")
        medias = {nombre: valores.mean() for nombre, valores in grupos.items()}
        mayor = max(medias, key=medias.get)
        menor = min(medias, key=medias.get)
        nombre_prop = nombres[prop4]
        unidad = unidades[prop4]

        if p < 0.05:
            st.success(
                f"**Sí, la condición de procesamiento parece influir en {nombre_prop}.** "
                f"Las diferencias entre grupos son demasiado grandes para explicarse "
                f"solo por casualidad. El promedio más alto es el de **{mayor}** "
                f"({medias[mayor]:.1f} {unidad}) y el más bajo el de **{menor}** "
                f"({medias[menor]:.1f} {unidad}).")
        else:
            st.info(
                f"**No hay evidencia suficiente de que la condición afecte {nombre_prop}.** "
                f"Las diferencias que se ven entre grupos podrían deberse al azar. "
                f"Ojo: esto no prueba que no haya efecto, solo que estos datos no lo muestran.")

        # Gráfica de promedios por grupo (con desviación estándar)
        tabla = pd.DataFrame({
            "Condición": list(grupos.keys()),
            "Promedio": [v.mean() for v in grupos.values()],
            "Desv. estándar": [v.std() for v in grupos.values()],
            "n": [len(v) for v in grupos.values()],
        })
        fig = px.bar(tabla, x="Condición", y="Promedio", error_y="Desv. estándar",
                     color="Condición", text="n",
                     labels={"Promedio": f"{PROPIEDADES[prop4]} (promedio)"})
        fig.update_traces(texttemplate="n = %{text}", textposition="inside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Las barras muestran el promedio de cada grupo; las líneas, la dispersión "
                   "(desviación estándar). Si los grupos se traslapan mucho, la diferencia es poco clara.")

        # ---------- Advertencias para concluir correctamente ----------
        st.markdown("#### ⚠️ Antes de concluir")
        if p_lev < 0.05:
            st.warning("Las varianzas de los grupos son muy distintas (Levene p < 0.05): "
                       "el ANOVA clásico es menos confiable. Apóyate más en Kruskal-Wallis.")
        if (p < 0.05) != (p_kw < 0.05):
            st.warning("ANOVA y Kruskal-Wallis no coinciden. La conclusión no es sólida; "
                       "preséntala con cautela.")
        st.markdown("""
- El ANOVA dice que **al menos un grupo** es distinto, pero **no dice cuáles** ni prueba la causa.
- Cada condición incluye aceros con distinto %C, y el carbono también cambia las propiedades. 
  Parte de la diferencia puede deberse al carbono y no al proceso.
- Los datos son valores nominales de manual, no mediciones repetidas.
- Que sea estadísticamente significativo no significa que importe en ingeniería: 
  mira cuántos {} cambia el valor.
        """.format(unidad))

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
