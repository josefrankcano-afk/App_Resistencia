# ============================================================
# LABORATORIO CIENTÍFICO FERROVIARIO | TGV DUPLEX
# MODELO DE TRACCIÓN - RESISTENCIA
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import brentq

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Laboratorio Científico Ferroviario",
    layout="wide"
)

st.title("Laboratorio Científico Ferroviario")
st.subheader("Experimento 01 | Modelo de tracción y resistencia")
st.write("Simulación del comportamiento dinámico longitudinal del TGV Duplex.")

# ============================================================
# 2. PARÁMETROS DE ENTRADA
# ============================================================

st.sidebar.header("Material rodante")

M_t = st.sidebar.number_input("Masa total [t]", 1.0, 1000.0, 424.0)
M_adh_t = st.sidebar.number_input("Masa adherente [t]", 1.0, 1000.0, 424.0)
P_kW = st.sidebar.number_input("Potencia [kW]", 100.0, 20000.0, 8800.0)
phi0 = st.sidebar.number_input("Adherencia inicial", 0.01, 1.0, 0.30)

st.sidebar.header("Coeficientes de Davis")

a = st.sidebar.number_input("Coeficiente a", value=0.637, format="%.5f")
b = st.sidebar.number_input("Coeficiente b", value=0.00755, format="%.5f")
c = st.sidebar.number_input("Coeficiente c", value=0.000126, format="%.6f")

st.sidebar.header("Infraestructura")

R_m = st.sidebar.number_input("Radio de curva [m]", min_value=1.0, value=2000.0)
i_permille = st.sidebar.number_input("Pendiente [‰]", value=10.0)

st.sidebar.header("Simulación")

v_min = st.sidebar.number_input("Velocidad mínima [km/h]", min_value=0.1, value=5.0)
v_max = st.sidebar.number_input("Velocidad máxima [km/h]", min_value=0.2, value=350.0)
n = st.sidebar.slider("Número de puntos", 100, 5000, 2000, 100)

if v_min >= v_max:
    st.error("La velocidad mínima debe ser menor que la máxima.")
    st.stop()

if M_adh_t > M_t:
    st.error("La masa adherente no puede superar la masa total.")
    st.stop()

g = 9.81

# ============================================================
# 3. FUNCIONES MATEMÁTICAS
# ============================================================

def F_pot(v):
    return 3.6 * P_kW / v

def phi(v):
    return phi0 * (0.2155 + 33 / (v + 42))

def F_adh(v):
    return phi(v) * M_adh_t * g

def F_trac(v):
    return np.minimum(F_pot(v), F_adh(v))

def R_davis(v):
    return (a + b*v + c*v**2) * M_t * 0.01

R_curva = (600 / R_m) * M_t * 0.01
R_pend = i_permille * M_t * 0.01

def R_total(v):
    return R_davis(v) + R_curva + R_pend

def F_exc(v):
    return F_trac(v) - R_total(v)

# ============================================================
# 4. CÁLCULO DEL PUNTO DE EQUILIBRIO
# ============================================================

v = np.linspace(v_min, v_max, n)
f = F_exc(v)

indices = np.where(f[:-1] * f[1:] < 0)[0]
raices = list(v[np.isclose(f, 0, atol=1e-9, rtol=0)])

for j in indices:
    raices.append(
        brentq(
            F_exc, v[j], v[j+1],
            xtol=1e-12, rtol=1e-14
        )
    )

raices = sorted(set(round(float(r), 10) for r in raices))
v_eq = max(raices) if raices else None

# ============================================================
# 5. RESULTADOS DEL EQUILIBRIO
# ============================================================

st.header("Resultados de la simulación")

if v_eq is not None:

    F_eq = float(F_trac(v_eq))
    R_eq = float(R_total(v_eq))
    error = F_eq - R_eq

    regimen = (
        "POTENCIA"
        if F_pot(v_eq) < F_adh(v_eq)
        else "ADHERENCIA"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Velocidad de equilibrio", f"{v_eq:.4f} km/h")
    col2.metric("Tracción", f"{F_eq:.4f} kN")
    col3.metric("Resistencia", f"{R_eq:.4f} kN")
    col4.metric("Régimen", regimen)

    st.write(f"**Diferencia residual:** {error:.3e} kN")

else:
    st.warning("No se encontró equilibrio en el intervalo analizado.")

# ============================================================
# 6. TABLA DE RESULTADOS
# ============================================================

# Incorporar la velocidad exacta de equilibrio
v_tabla = np.sort(np.append(v, v_eq)) if v_eq is not None else v

tabla = pd.DataFrame({
    "v_km_h": v_tabla,
    "phi": phi(v_tabla),
    "F_pot_kN": F_pot(v_tabla),
    "F_adh_kN": F_adh(v_tabla),
    "F_disp_kN": F_trac(v_tabla),
    "R_Davis_kN": R_davis(v_tabla),
    "R_total_kN": R_total(v_tabla),
    "F_exc_kN": F_exc(v_tabla)
})

# Identificar la fila de equilibrio
tabla["Equilibrio"] = False

if v_eq is not None:
    indice_eq = (tabla["v_km_h"] - v_eq).abs().idxmin()
    tabla.loc[indice_eq, "Equilibrio"] = True

# Estilo de resaltado
def resaltar_equilibrio(fila):
    if fila["Equilibrio"]:
        return [
            "background-color: #FFE066; color: #000000; "
            "font-weight: bold;"
        ] * len(fila)

    return [""] * len(fila)

st.subheader("Tabla de resultados")

# Tabla completa con resaltado
tabla_estilizada = (
    tabla.style
    .apply(resaltar_equilibrio, axis=1)
    .format(precision=4, subset=tabla.columns[:-1])
    .hide(subset=["Equilibrio"], axis="columns")
)

st.dataframe(
    tabla_estilizada,
    use_container_width=True,
    hide_index=True,
    height=550
)

# ============================================================
# RESULTADO DESTACADO DEL EQUILIBRIO
# ============================================================

if v_eq is not None:

    st.markdown("#### Punto de equilibrio")

    fila_equilibrio = tabla.loc[
        tabla["Equilibrio"]
    ].drop(columns=["Equilibrio"])

    st.table(
        fila_equilibrio.style
        .format(precision=4)
        .set_properties(**{
            "background-color": "#FFE066",
            "color": "#000000",
            "font-weight": "bold"
        })
        .hide(axis="index")
    )

    st.caption(
        f"Equilibrio longitudinal: {v_eq:.8f} km/h | "
        f"Tracción: {F_eq:.8f} kN | "
        f"Resistencia: {R_eq:.8f} kN"
    )

else:
    st.info("No se encontró un punto de equilibrio en el intervalo.")

# ============================================================
# DESCARGAR RESULTADOS
# ============================================================

st.download_button(
    "Descargar resultados CSV",
    tabla.drop(columns=["Equilibrio"])
         .to_csv(index=False)
         .encode("utf-8-sig"),
    file_name="resultados_tgv_duplex.csv",
    mime="text/csv"
)
# ============================================================
# 7. GRÁFICA INTERACTIVA
# ============================================================

st.subheader("Curvas de tracción y resistencia")

fig = go.Figure()

curvas = [
    (F_pot(v), "Límite por potencia"),
    (F_adh(v), "Límite por adherencia"),
    (F_trac(v), "Tracción disponible"),
    (R_davis(v), "Resistencia Davis"),
    (R_total(v), "Resistencia total")
]

for y, nombre in curvas:
    fig.add_trace(go.Scatter(
        x=v,
        y=y,
        mode="lines",
        name=nombre,
        hovertemplate=(
            f"<b>{nombre}</b><br>"
            "v = %{x:.3f} km/h<br>"
            "F = %{y:.3f} kN<extra></extra>"
        )
    ))

if v_eq is not None:

    fig.add_trace(go.Scatter(
        x=[v_eq],
        y=[F_eq],
        mode="markers",
        marker=dict(size=13, color="cyan"),
        name="Punto de equilibrio",
        hovertemplate=(
            "<b>EQUILIBRIO</b><br>"
            f"v = {v_eq:.8f} km/h<br>"
            f"F = {F_eq:.8f} kN<br>"
            f"R = {R_eq:.8f} kN<extra></extra>"
        )
    ))

    fig.add_vline(
        x=v_eq,
        line_dash="dash",
        annotation_text=f"Equilibrio = {v_eq:.2f} km/h"
    )

fig.update_layout(
    title="Curvas de tracción y resistencia | TGV Duplex",
    xaxis_title="Velocidad [km/h]",
    yaxis_title="Fuerza [kN]",
    hovermode="x unified",
    template="plotly_white",
    height=650
)

fig.update_xaxes(range=[0, max(400, v_max)])
fig.update_yaxes(rangemode="tozero")

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. CONSULTA DE VELOCIDAD
# ============================================================

st.subheader("Análisis de un punto de operación")

v_consulta = st.slider(
    "Velocidad de consulta [km/h]",
    float(v_min),
    float(v_max),
    float((v_min + v_max) / 2)
)

c1, c2, c3 = st.columns(3)

c1.metric("Tracción disponible", f"{F_trac(v_consulta):.3f} kN")
c2.metric("Resistencia total", f"{R_total(v_consulta):.3f} kN")
c3.metric("Esfuerzo excedente", f"{F_exc(v_consulta):.3f} kN")