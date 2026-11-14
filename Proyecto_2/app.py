"""
Explorador de aceros al carbono - Proyecto 2 (Ingeniería de materiales)

Cómo correrlo:
    pip install -r requirements.txt
    streamlit run app.py

El CSV debe estar en la misma carpeta que este archivo.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from pathlib import Path

ARCHIVO = next(Path(__file__).parent.glob("*.csv"))
NAN = float("nan")

PROPIEDADES = {
    "UTS": "Resistencia última a la tensión, UTS (MPa)",
    "YS": "Esfuerzo de fluencia, YS (MPa)",
    "Dureza": "Dureza (HB)",
    "Elongacion": "Elongación (%)",
}
NOMBRES_CORTOS = {"UTS": "UTS", "YS": "YS", "Dureza": "dureza", "Elongacion": "elongación"}
UNIDADES = {"UTS": "MPa", "YS": "MPa", "Dureza": "HB", "Elongacion": "%"}

# Rangos razonables para aceptar un dato nuevo (evita errores de captura)
LIMITES = {"UTS": (100, 2500), "YS": (50, 2500), "Dureza": (50, 700), "Elongacion": (1, 60)}


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
# 1b. AYUDAS PARA AGREGAR MATERIALES NUEVOS
# ============================================================

# --- INICIO AYUDAS ---
def validar_nuevo(sae, c, valores):
    """Devuelve la lista de errores del registro nuevo (vacía si es válido)."""
    errores = []
    if sae == "":
        errores.append("Escribe el grado SAE o el nombre del acero.")

    if c is None:
        errores.append("El % de carbono es obligatorio: sin él, el dato no entra a las "
                       "gráficas de carbono ni a la predicción.")
    elif not 0.01 <= c <= 1.10:
        errores.append("El %C debe estar entre 0.01 y 1.10 (aceros al carbono).")

    dados = {k: v for k, v in valores.items() if v is not None}
    if len(dados) < 2:
        errores.append(f"Captura al menos 2 de las 4 propiedades (ahora hay {len(dados)}). "
                       "Con menos datos, el registro casi no aporta a ninguna comparación "
                       "ni al ANOVA.")
    for nombre, valor in dados.items():
        bajo, alto = LIMITES[nombre]
        if not bajo <= valor <= alto:
            errores.append(f"{nombre} = {valor} {UNIDADES[nombre]} está fuera del rango "
                           f"razonable ({bajo} a {alto} {UNIDADES[nombre]}).")
    if "UTS" in dados and "YS" in dados and dados["YS"] > dados["UTS"]:
        errores.append("El YS no puede ser mayor que la UTS.")
    return errores


def es_duplicado(df, sae, condicion, valores):
    """True si ya existe un registro con el mismo acero, condición y valores."""
    candidatos = df[(df["SAE"].str.lower() == sae.lower()) & (df["Condicion"] == condicion)]
    for nombre, valor in valores.items():
        if valor is not None:
            candidatos = candidatos[candidatos[nombre] == valor]
    return len(candidatos) > 0


def resumen_impacto(df_todo, fila):
    """Explica en qué análisis entra (o no) el registro recién agregado."""
    condicion = fila["Condicion"]
    grupo = df_todo[df_todo["Condicion"] == condicion]
    lineas = []

    for nombre, nombre_corto in NOMBRES_CORTOS.items():
        if pd.isna(fila[nombre]):
            continue
        n = int(grupo[nombre].notna().sum())
        if n >= 3:
            lineas.append(f"✅ **ANOVA de {nombre_corto}:** el dato se suma al grupo "
                          f"«{condicion}» ({n} registros).")
        else:
            lineas.append(f"⚠️ **ANOVA de {nombre_corto}:** el grupo «{condicion}» solo tiene "
                          f"{n} registro(s) con este dato; el ANOVA exige al menos 3, así que "
                          f"ese grupo todavía no entra a la prueba.")

    lineas.append("✅ **%C vs propiedades y Procesamiento:** el dato aparece en las gráficas "
                  "(si los filtros de la barra lateral lo incluyen).")

    n_modelo = len(grupo.dropna(subset=["C_prom", "UTS", "YS"]))
    if pd.notna(fila["UTS"]) and pd.notna(fila["YS"]) and n_modelo > 3:
        lineas.append(f"✅ **Predicción:** entra al entrenamiento del modelo de «{condicion}» "
                      f"({n_modelo} registros).")
    else:
        lineas.append("⚠️ **Predicción:** no entra al modelo (necesita UTS y YS, y más de 3 "
                      "registros completos en esa condición).")
    return "\n\n".join(lineas)
# --- FIN AYUDAS ---


# ============================================================
# 2. CONFIGURACIÓN DE LA PÁGINA Y BARRA LATERAL
# ============================================================

st.set_page_config(page_title="Aceros al carbono", page_icon="🔩", layout="wide")
st.title("🔩 Explorador de aceros al carbono")
st.caption("Composición química, procesamiento y propiedades mecánicas")

# Datos originales + materiales que el usuario agregó en esta sesión
df_base = cargar_datos().assign(Origen="Dataset original")
df_base["SAE"] = df_base["SAE"].astype(str)

if "nuevos" not in st.session_state:
    st.session_state["nuevos"] = []
if "form_n" not in st.session_state:
    st.session_state["form_n"] = 0

if st.session_state["nuevos"]:
    df = pd.concat([df_base, pd.DataFrame(st.session_state["nuevos"])], ignore_index=True)
else:
    df = df_base

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
    "🤖 Predicción",
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

    with st.expander("Decisiones de limpieza"):
        st.markdown("""
- **Agrupación de condiciones:** Las 46 etiquetas originales se simplificaron en categorías (Recocido, Normalizado, Estirado en frío, etc.) y se corrigieron errores de captura ("Cold Orawn", "Hot rotted"). Las combinaciones con estirado en frío se separaron como categorías propias para no mezclar efectos.
- **Separador decimal:** Se reemplazaron las comas por puntos en "Reducción" para que Python las lea como números.
- **Dureza:** Los registros en Rockwell B (HRB) se dejaron vacíos para mantener una sola escala (Brinell, HB).
- **Contenido de carbono (%C):** Se corrigió un error de escala evidente (C = 15 en el acero 12L14, que corresponde a 0.15) y se usó el punto medio entre el mínimo y el máximo como valor representativo. Cuando falta el mínimo, se usa solo el máximo.
- **ANOVA:** Solo se incluyen grupos con al menos 3 registros, por lo que las categorías muy pequeñas quedan fuera.
        """)

    # ---------- Agregar un material nuevo ----------
    st.markdown("---")
    st.subheader("➕ Agregar un material nuevo")

    aviso = st.session_state.pop("aviso", None)
    if aviso:
        st.success(aviso["titulo"])
        st.markdown(aviso["detalle"])

    with st.expander("Formulario para registrar un acero"):
        st.markdown("""
**Reglas mínimas para que el dato sirva en el análisis:**
- **Obligatorios:** nombre o grado del acero, condición de procesamiento y %C. Sin %C el dato
  no puede entrar a las gráficas de carbono ni a la predicción.
- **Propiedades:** captura al menos **2 de las 4** (UTS, YS, dureza, elongación). El resto puede
  quedar vacío.
- **Por qué:** el ANOVA compara promedios por condición y necesita al menos **3 registros con la
  propiedad medida** en cada grupo. Un registro casi vacío no aporta a ningún grupo y solo
  ensucia las gráficas. Al guardar, te indico en qué análisis entra tu dato.
- Los valores deben estar en rangos razonables y el YS no puede superar a la UTS.
        """)

        n = st.session_state["form_n"]  # al cambiar, el formulario se limpia
        with st.form(f"form_nuevo_{n}"):
            f1, f2, f3 = st.columns(3)
            sae_nuevo = f1.text_input("Grado SAE o nombre del acero *",
                                      placeholder="ej. 1045", key=f"sae_{n}")
            cond_nueva = f2.selectbox("Condición de procesamiento *",
                                      sorted(df["Condicion"].unique()), key=f"cond_{n}")
            c_nuevo = f3.number_input("% de carbono *", min_value=0.0, value=None,
                                      step=0.01, format="%.3f", key=f"c_{n}")

            g1, g2, g3, g4 = st.columns(4)
            uts_n = g1.number_input("UTS (MPa)", min_value=0.0, value=None,
                                    step=10.0, key=f"uts_{n}")
            ys_n = g2.number_input("YS (MPa)", min_value=0.0, value=None,
                                   step=10.0, key=f"ys_{n}")
            dureza_n = g3.number_input("Dureza (HB)", min_value=0.0, value=None,
                                       step=5.0, key=f"dur_{n}")
            elong_n = g4.number_input("Elongación (%)", min_value=0.0, value=None,
                                      step=1.0, key=f"elo_{n}")
            enviar = st.form_submit_button("Agregar material")

        if enviar:
            sae = sae_nuevo.strip()
            valores = {"UTS": uts_n, "YS": ys_n, "Dureza": dureza_n, "Elongacion": elong_n}
            errores = validar_nuevo(sae, c_nuevo, valores)

            if errores:
                st.error("No se agregó el material. Corrige lo siguiente:")
                for e in errores:
                    st.write(f"- {e}")
            elif es_duplicado(df, sae, cond_nueva, valores):
                st.warning("Ya existe un registro con ese acero, esa condición y esos valores. "
                           "Agregarlo no cambiaría ningún resultado, así que no se guardó.")
            else:
                fila = {
                    "SAE": sae,
                    "Condicion_original": "Agregado por el usuario",
                    "UTS": NAN if uts_n is None else uts_n,
                    "YS": NAN if ys_n is None else ys_n,
                    "Elongacion": NAN if elong_n is None else elong_n,
                    "Reduccion": NAN,
                    "Dureza": NAN if dureza_n is None else dureza_n,
                    "C_min": c_nuevo, "C_max": c_nuevo, "C_prom": c_nuevo,
                    "Condicion": cond_nueva,
                    "Origen": "Agregado por el usuario",
                }
                df_todo = pd.concat([df, pd.DataFrame([fila])], ignore_index=True)
                st.session_state["aviso"] = {
                    "titulo": f"✅ Material «{sae}» agregado en la condición «{cond_nueva}».",
                    "detalle": "**Impacto en el análisis:**\n\n" + resumen_impacto(df_todo, fila),
                }
                st.session_state["nuevos"].append(fila)
                st.session_state["form_n"] += 1
                st.rerun()

    if st.session_state["nuevos"]:
        nuevos_df = pd.DataFrame(st.session_state["nuevos"])
        st.info(f"Has agregado **{len(nuevos_df)}** registro(s) en esta sesión. "
                "Solo viven en esta sesión: se pierden al recargar la página.")
        b1, b2 = st.columns(2)
        b1.download_button("⬇️ Descargar mis registros (CSV)",
                           nuevos_df.to_csv(index=False).encode("utf-8"),
                           file_name="materiales_agregados.csv", mime="text/csv")
        if b2.button("🗑️ Borrar registros agregados"):
            st.session_state["nuevos"] = []
            st.rerun()

    st.markdown("---")
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

    st.info("💡 **Interpretación de ingeniería:** Al aumentar el %C aumenta la fracción de perlita "
            "(laminillas de ferrita y cementita) en la microestructura. La cementita y las interfaces "
            "ferrita/cementita obstaculizan el movimiento de las dislocaciones, por lo que suben la UTS, "
            "el YS y la dureza, mientras que la elongación disminuye. En los datos la tendencia es clara "
            "pero con dispersión (correlación moderada), porque en una misma gráfica se mezclan aceros con "
            "distinto procesamiento. Usa el filtro de condición para ver la tendencia dentro de un solo proceso.")

# ============================================================
# 5. PESTAÑA: PROCESAMIENTO
# ============================================================
with tabs[2]:
    st.subheader("Comparación entre condiciones de procesamiento")

    with st.expander("📘 ¿En qué consiste cada proceso? (explicación sencilla)"):
        st.markdown("""
Un acero no solo depende de **qué lleva** (su composición), sino de **cómo se trabajó**.
Calentarlo, enfriarlo o deformarlo cambia la forma en que se acomodan sus cristales
(su *microestructura*), y eso cambia qué tan fuerte, duro o flexible es.

**Dos ideas para recordar:**
- **Enfriar más rápido → acero más duro y resistente, pero más frágil.**
- **Deformarlo en frío → acero más duro y resistente, pero menos flexible.**

---

**🔥 Laminado en caliente**
El acero se calienta al rojo vivo (alrededor de 1000 °C o más) y se hace pasar entre rodillos
que lo aplastan hasta darle forma de barra, placa o perfil. Estando tan caliente, el metal se
deforma con facilidad y sus cristales se reacomodan solos, así que **no queda endurecido**.
Es el proceso más común y barato; deja una superficie algo rugosa y medidas menos precisas.
*Analogía: estirar una masa tibia con un rodillo de cocina.*

**🧵 Estirado en frío**
Una barra ya formada se jala, **a temperatura ambiente**, a través de un agujero más pequeño
(un dado). Al no calentarse, los cristales quedan "apretados y enredados": el acero sale más
**duro y resistente**, pero **pierde parte de su capacidad de estirarse**. Además queda con
acabado liso y medidas muy precisas.
*Analogía: doblar un clip de metal varias veces: se vuelve más rígido y al final se rompe.*

**🛋️ Recocido**
Se calienta el acero (en estos datos, entre 800 y 915 °C), se mantiene un tiempo y se
**enfría muy lentamente**, por ejemplo dejándolo dentro del horno apagado. El resultado es un
acero **blando y flexible**, sin tensiones internas y fácil de maquinar.
*Analogía: dejar que un músculo tenso se relaje por completo.*

**🌬️ Normalizado**
Parecido al recocido, pero calentando un poco más y **enfriando al aire libre**, que es más
rápido que dentro de un horno. Ese enfriamiento más rápido forma **granos más finos y
uniformes**, lo que da más resistencia que el recocido sin volver el acero muy frágil.
*Analogía: la misma receta que el recocido, pero sacándolo a enfriar a la mesa.*

**💧 Templado y revenido** (dos pasos)
1. **Temple:** se calienta el acero (800 a 885 °C en estos datos) y se **enfría de golpe** en
   agua o aceite. Se forma una estructura llamada martensita, **muy dura pero frágil**.
2. **Revenido:** se recalienta a una temperatura mucho menor (540 °C en estos datos) y se deja
   enfriar. Así se recupera algo de flexibilidad sin perder demasiada resistencia.

Es el tratamiento que logra las resistencias más altas de la base de datos.
*Analogía: primero se "congela" una estructura muy dura y rígida, y luego se "alivia" un poco.*

**🔗 Combinaciones**
- **Recocido + estirado en frío:** primero se ablanda el acero (en 12 de los 21 registros con
  *recocido esferoidizante*, donde el carbono se acomoda en pequeñas esferas que facilitan
  deformarlo) y luego se estira en frío para darle medidas precisas y subir su resistencia.
- **Normalizado + estirado en frío:** lo mismo, partiendo de un acero normalizado. Tiene solo
  2 registros, por lo que no alcanza para análisis estadísticos.
        """)

        resumen_procesos = pd.DataFrame({
            "Proceso": ["Laminado en caliente", "Estirado en frío", "Recocido",
                        "Normalizado", "Templado y revenido"],
            "¿Se calienta?": ["Sí, muy caliente", "No (temperatura ambiente)", "Sí",
                              "Sí", "Sí (y luego se recalienta)"],
            "¿Cómo se enfría o se trabaja?": ["Rodillos que lo aplastan", "Se jala por un dado",
                                              "Enfriamiento muy lento (horno)",
                                              "Enfriamiento al aire", "Enfriamiento muy rápido (agua o aceite)"],
            "Efecto típico": ["Resistencia media, buena ductilidad", "Más resistente, menos dúctil",
                              "El más blando y dúctil", "Más resistente que el recocido",
                              "El más duro y resistente"],
        })
        st.dataframe(resumen_procesos, use_container_width=True, hide_index=True)

    prop2 = st.selectbox("Propiedad ", list(PROPIEDADES),
                         format_func=PROPIEDADES.get, key="prop_proc")
    fig = px.box(dff, x="Condicion", y=prop2, color="Condicion", points="all",
                 hover_data=["SAE"], labels={prop2: PROPIEDADES[prop2]})
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    resumen = dff.groupby("Condicion")[prop2].agg(["count", "mean", "std"]).round(1)
    resumen.columns = ["n", "media", "desv. estándar"]
    st.dataframe(resumen, use_container_width=True)

    st.info("💡 **Interpretación de ingeniería:** El procesamiento define la microestructura final. "
            "El **estirado en frío** endurece por deformación (multiplicación y enredo de dislocaciones): "
            "sube el YS y baja la elongación. En estos datos la UTS casi no cambia respecto al laminado en "
            "caliente porque el grupo estirado en frío tiene, en promedio, menos %C (variable confusora). "
            "El **recocido** produce granos grandes y elimina tensiones residuales, dando aceros blandos y "
            "los más dúctiles. El **normalizado**, al enfriar al aire, refina el tamaño de grano "
            "(endurecimiento de Hall-Petch) y logra mayor resistencia que el recocido con buena ductilidad. "
            "El **templado y revenido** alcanza las resistencias más altas por la martensita revenida, "
            "a costa de menor ductilidad.")

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

    st.warning("⚠️ **Limitaciones del análisis:** La teoría indica que el normalizado produce mayor "
               "resistencia por el afino de grano, pero esta comparación tiene límites. No estamos "
               "comparando el *mismo* grado SAE con distinto tratamiento, sino conjuntos diferentes de "
               "aceros en cada categoría, así que parte de la diferencia puede deberse a la composición. "
               "Aquí ambos grupos tienen un tamaño similar (unos 20 registros) y un %C promedio casi igual, "
               "lo que ayuda, pero no elimina el problema. Además no se conocen las temperaturas ni los "
               "tamaños de sección de cada tratamiento, y los valores son nominales de manual, no mediciones repetidas.")

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
    unidad = UNIDADES[prop4]

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
        st.markdown(f"""
- El ANOVA dice que **al menos un grupo** es distinto, pero **no dice cuáles** ni prueba la causa.
- Cada condición incluye aceros con distinto %C, y el carbono también cambia las propiedades.
  Parte de la diferencia puede deberse al carbono y no al proceso.
- Los datos son valores nominales de manual, no mediciones repetidas.
- Que sea estadísticamente significativo no significa que importe en ingeniería:
  mira cuántos {unidad} cambia el valor.
        """)

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

# ============================================================
# 9. PESTAÑA: MODELO PREDICTIVO
# ============================================================
with tabs[6]:
    st.subheader("🤖 Predicción de propiedades mecánicas")
    st.write("Estima la UTS y el YS de un acero al carbono a partir de su %C "
             "y su condición de procesamiento.")

    df_ml = df.dropna(subset=["C_prom", "Condicion", "UTS", "YS"])

    col1, col2 = st.columns(2)
    cond_pred = col1.selectbox("Condición de procesamiento",
                               sorted(df_ml["Condicion"].unique()), key="ml_cond")
    c_pred = col2.number_input("% de carbono (ej. 0.45 para un 1045)",
                               min_value=0.05, max_value=1.10, value=0.45, step=0.01)

    df_train = df_ml[df_ml["Condicion"] == cond_pred]

    if len(df_train) > 3:
        X = df_train[["C_prom"]]
        modelo_uts = LinearRegression().fit(X, df_train["UTS"])
        modelo_ys = LinearRegression().fit(X, df_train["YS"])

        nuevo = pd.DataFrame({"C_prom": [c_pred]})
        uts_pred = modelo_uts.predict(nuevo)[0]
        ys_pred = modelo_ys.predict(nuevo)[0]
        r2_uts = modelo_uts.score(X, df_train["UTS"])
        r2_ys = modelo_ys.score(X, df_train["YS"])

        st.success(f"**Resultados estimados para %C = {c_pred:.2f} y condición = {cond_pred}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Esfuerzo máximo (UTS)", f"{uts_pred:.1f} MPa")
        c2.metric("Límite elástico (YS)", f"{ys_pred:.1f} MPa")
        c3.metric("Registros usados", len(df_train))

        # ---------- Qué tan confiable es ----------
        st.write(f"Ajuste del modelo (R²): UTS = **{r2_uts:.2f}**, YS = **{r2_ys:.2f}**")
        if min(r2_uts, r2_ys) < 0.5:
            st.warning("El %C explica poco de la variación en esta condición (R² bajo). "
                       "Los aceros de este grupo difieren en otros aspectos (aleantes, "
                       "tratamiento exacto), así que la predicción es muy aproximada.")
        c_min, c_max = df_train["C_prom"].min(), df_train["C_prom"].max()
        if c_pred < c_min or c_pred > c_max:
            st.warning(f"El %C ingresado está fuera del rango de los datos de esta condición "
                       f"({c_min:.2f} a {c_max:.2f}). Es una extrapolación y no es confiable.")

        # ---------- Gráfica de datos, modelo y predicción ----------
        fig = go.Figure()
        for nombre, modelo in (("UTS", modelo_uts), ("YS", modelo_ys)):
            fig.add_trace(go.Scatter(x=df_train["C_prom"], y=df_train[nombre],
                                     mode="markers", name=f"{nombre} (datos)"))
            xs = [c_min, c_max]
            ys = modelo.predict(pd.DataFrame({"C_prom": xs}))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{nombre} (modelo)"))
        fig.add_trace(go.Scatter(x=[c_pred, c_pred], y=[uts_pred, ys_pred],
                                 mode="markers", name="Predicción",
                                 marker=dict(symbol="star", size=14)))
        fig.update_layout(xaxis_title="%C (punto medio)", yaxis_title="Esfuerzo (MPa)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption("Regresión lineal simple (una por condición) con el %C como única variable. "
                   "Se entrena con valores nominales de manual, no con mediciones repetidas. "
                   "Resultados aproximados, no para diseño.")
    else:
        st.error("No hay suficientes datos en esta condición para entrenar el modelo.")