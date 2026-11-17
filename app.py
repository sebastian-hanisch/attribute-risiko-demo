"""Mehrere Ziele unter Risiko: Skalierungskonstanten, Form und Kopplung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zwei Ziele (Gewinn in GE, Servicegrad in %) im ausgedachten Kapazitätsfall und in Zufallsproblemen: Skalierungskonstanten k_i, additive und multiplikative Form,
Risiko innerhalb der Attribute und die Kopplung der Attribute bei gleichen Rändern. Das Zahlenbeispiel ist ein Modell, kein Fall.

Lauffähig mit: streamlit run app.py
"""

import math
from fractions import Fraction

import pandas as pd
import streamlit as st

import atr_constants as C
import atr_evaluation as ev
import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U
from atr_layout import dunkles_design, kopf_css
from atr_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from atr_visualization import build_coupling_bars, build_eu_bars, build_form_bars, build_rho_curves, build_risk_curve

st.set_page_config(page_title="Mehrere Ziele unter Risiko – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(hs):
    return ev.analyse_hand(hs)


@st.cache_data(show_spinner=False)
def _form_table(P, n, m, seed, sums):
    return SW.form_table(P, n, m, seed, sums)


@st.cache_data(show_spinner=False)
def _form_example(P, n, m, seed, sum_k):
    return SW.example_matrix(P, n, m, seed, sum_k)


@st.cache_data(show_spinner=False)
def _continuity(P, n, m, seed):
    return SW.continuity_probe(P, n, m, seed)


@st.cache_data(show_spinner=False)
def _risk_table(P, S, seed, form):
    return SW.risk_table(P, S, seed, form)


@st.cache_data(show_spinner=False)
def _risk_scenarios(P, seed, rho_norm, form):
    return SW.risk_by_scenarios(P, seed, rho_norm, form)


@st.cache_data(show_spinner=False)
def _coupling_table(P, n, seed, sums):
    return SW.coupling_table(P, n, seed, sums)


@st.cache_data(show_spinner=False)
def _coupling_example(P, n, seed, sum_k):
    return SW.coupling_example(P, n, seed, sum_k)


def _state(key):
    """Wert eines Reglers; die Standardwerte stehen vor dem ersten Zeichnen im Session State (init_session_state_defaults)."""
    return st.session_state[key]


def _sums_with(current):
    return tuple(sorted(set(C.SUM_K_VALUES) | {round(float(current), 2)}))


def _rho_label(rho):
    return "∞ (linear)" if math.isinf(rho) else f"{rho:g} GE".replace(".", ",")


def _cell_table(res):
    hs = res.hs
    rows = []
    for alt in SC.ALTS:
        for c in res.cells[alt]:
            rows.append({
                "Alternative": alt, "Nachfrage": c.demand, "Lieferlage": c.supply, "Wahrscheinlichkeit": ev.fr_short(c.p), "Gewinn (GE)": c.gain,
                "Servicegrad": f"{ev.de(100 * float(c.service), 0)} %", "u₁ (Gewinn)": ev.fr(c.u1, 3), "u₂ (Service)": ev.fr(c.u2, 3),
                "U multiplikativ": ev.fr(U.u2_mult(c.u1, c.u2, hs.k1, hs.k2), 3), "U additiv": ev.fr(U.u2_add(c.u1, c.u2, hs.k1, hs.k2), 3),
            })
    return pd.DataFrame(rows)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in atr_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Mehrere Ziele unter Risiko: Skalierungskonstanten, Form und Kopplung")
st.markdown(
    """
Haben die Ausgänge **zwei Ziele**, hier **Gewinn in GE** und **Servicegrad in %**, reicht ein Einzelnutzen nicht mehr: man braucht je Ziel einen Einzelnutzen $u_i$ und
**Skalierungskonstanten $k_i$**, und deren **Summe Σk** ist keine Formalität. Bei Σk = 1 ist die multiplikative Form die additive; sonst steckt in **c = 1 − Σk** eine
Wechselwirkung, und die multiplikative Form hängt dann nicht nur von den Rändern jedes Ziels ab, sondern von ihrer **Kopplung**. Die Demo zeigt das im Kapazitätsfall und
an Zufallsproblemen (Zähleinheit Matrizen bzw. Probleme).

**Die Auszahlungen, die Servicegrade und die $k_i$ sind ausgedacht: das Beispiel ist ein Modell, kein Fall und keine Messung an Menschen.** Alles hier ist Lehrbuchstoff
(Produktform, K, Skalierungskonstanten); die Vorführung mit Zähleinheit und die Kopplungsmessung sind eigene Nachrechnung in diesem Modell.

Weiter unten: die **Kernfragen**, drei weitere Experimente, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die Rechnung mit zwei Zielen", expanded=True):
    st.markdown(
        """
1. **Je Ziel ein Einzelnutzen $u_i \\in [0, 1]$**: schlechtester Wert 0, bester Wert 1. Beim Gewinn exponentiell mit Risikotoleranz ρ in GE (ρ = ∞: linear), beim Servicegrad linear.
2. **Skalierungskonstanten $k_i$** = Nutzen der Ecke „Ziel i am besten, alle anderen am schlechtesten“ (bei der Erhebung per Lotterie: Indifferenz bei $p = k_i$). Hier sind sie gesetzt; die Erhebung selbst zeigt [`nutzen-erheben-demo`](https://github.com/sebastian-hanisch/nutzen-erheben-demo).
3. **Form**: additiv $U = \\sum_i (k_i / \\sum_j k_j)\\, u_i$ oder multiplikativ $1 + K\\,U = \\prod_i (1 + K\\,k_i\\,u_i)$. Der Erwartungsnutzen läuft über die Zellen der gemeinsamen Verteilung.
4. **Warum Σk zählt:** bei Σk = 1 ist K = 0 und beide Formen sind gleich; sonst gibt es den Kreuzterm $c\\,u_1 u_2$ mit $c = 1 - k_1 - k_2$, und der reagiert auf die Kopplung der Ziele.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Skalierungskonstanten (Kapazitätsfall)**")
    k1 = st.slider("k₁ (Gewinn)", *bounds("k1_slider"), step=C.K_STEP, key="k1_slider", help="Nutzen der Ecke „Gewinn am besten, Servicegrad am schlechtesten“.")
    k2 = st.slider("k₂ (Servicegrad)", *bounds("k2_slider"), step=C.K_STEP, key="k2_slider", help="Nutzen der Ecke „Servicegrad am besten, Gewinn am schlechtesten“.")
    st.caption(f"Σk = {ev.de(round(k1 + k2, 2), 2)}, c = 1 − Σk = {ev.de(round(1 - k1 - k2, 2), 2, True)}.")
    st.markdown("**Zufall (Experimente 2 bis 4)**")
    seed = st.number_input("Zufalls-Seed der Zufallsprobleme", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

# --- Experiment 1 ---------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.subheader("🔬 Experiment 1: Von Hand, zwei Ziele und ein Nutzen")
st.caption("Kapazitätsfall (ausgedacht): G (groß bauen) gewinnt 200 GE bei hoher und verliert 100 GE bei niedriger Nachfrage, K (klein bauen oder mieten) gewinnt sicher 30 GE. "
           "Zweites Ziel: der Servicegrad (100 % bei normaler Lieferlage, im Engpass weniger). Nachfrage hoch und Engpass haben je die Wahrscheinlichkeit ½; ihre Kopplung q = P(hoch und Engpass) ist der Regler.")
c1, c2, c3, c4 = st.columns(4)
q = c1.slider("Kopplung q = P(hoch und Engpass)", *bounds("q_slider"), step=C.Q_STEP, key="q_slider",
              help="q = ¼: unabhängig; q = 0: Engpass nur bei niedriger Nachfrage; q = ½: hohe Nachfrage immer mit Engpass. Die Ränder (je ½) bleiben gleich.")
rho = c2.select_slider("Risikotoleranz ρ des Gewinns", options=C.RHO_OPTIONS, key="rho_select", format_func=_rho_label,
                       help="ρ = ∞: linearer Gewinnnutzen (Brüche). Kleinere ρ: stärker risikoscheu. Der Gewinn allein kippt bei 555,8 GE.")
s_g = c3.slider("Servicegrad von G im Engpass (%)", *bounds("sg_slider"), step=C.SERVICE_STEP, key="sg_slider")
s_k = c4.slider("Servicegrad von K im Engpass (%)", *bounds("sk_slider"), step=C.SERVICE_STEP, key="sk_slider")

hs = ev.HandSettings(U.as_fraction(k1), U.as_fraction(k2), U.as_fraction(q), float(rho), Fraction(int(s_g), 100), Fraction(int(s_k), 100))
hand = _hand(hs)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Multiplikativ: EU(G)", ev.fr_short(hand.eu[("G", "mult")]), delta=f"G − K = {ev.de(float(hand.d_mult), 4, True)}", delta_color="off")
m2.metric("Multiplikativ: EU(K)", ev.fr_short(hand.eu[("K", "mult")]), delta=f"wählt {ev.winner(hand.d_mult)}", delta_color="off")
m3.metric("Additiv: EU(G)", ev.fr_short(hand.eu[("G", "add")]), delta=f"G − K = {ev.de(float(hand.d_add), 4, True)}", delta_color="off")
m4.metric("Additiv: EU(K)", ev.fr_short(hand.eu[("K", "add")]), delta=f"wählt {ev.winner(hand.d_add)}", delta_color="off")
st.plotly_chart(build_eu_bars(hand), width="stretch", key="eu_bars")
st.caption("Balken: Erwartungsnutzen je Alternative und Form; bei der multiplikativen Form getrennt in den Randterm $k_1 E u_1 + k_2 E u_2$ und den Kreuzterm $c \\cdot E[u_1 u_2]$ (hängt von der Kopplung ab).")
st.dataframe(_cell_table(hand), hide_index=True, width="stretch")
kind, text = ev.verdict_hand(hand)
getattr(st, kind)(text)

# --- Kernfragen -----------------------------------------------------------------------------------------------------------------------------

sum_form = round(float(_state("sumk_form_slider")), 2)
sum_kopp = round(float(_state("sumk_kopp_slider")), 2)
P_form, n_form, m_form = int(_state("p_form_slider")), int(_state("n_form_slider")), int(_state("m_slider"))
P_risk, S_risk = int(_state("p_risk_slider")), int(_state("s_slider"))
P_kopp, n_kopp = int(_state("p_kopp_slider")), int(_state("n_kopp_slider"))
seed_i = int(seed)
rho_norm, form_risk = float(_state("rho_norm_select")), _state("form_risk_radio")

with st.spinner("Rechne die Zufallsprobleme (meist unter einer Sekunde) …"):
    form_rows = _form_table(P_form, n_form, m_form, seed_i, _sums_with(sum_form))
    risk_rows = _risk_table(P_risk, S_risk, seed_i, form_risk)
    kopp_rows = _coupling_table(P_kopp, n_kopp, seed_i, _sums_with(sum_kopp))
form_row = next(r for r in form_rows if abs(r.sum_k - sum_form) < 1e-9)
kopp_row = next(r for r in kopp_rows if abs(r.sum_k - sum_kopp) < 1e-9)
risk_row = next(r for r in risk_rows if abs(r.rho_norm - rho_norm) < 1e-9)

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Warum genügt ein Nutzen nicht mehr, und was sagt Σk?** Jedes Ziel braucht einen Einzelnutzen $u_i$ und eine Konstante $k_i$; die Summe entscheidet, ob ein Kreuzterm entsteht.")
kind, text = ev.kern_one(hand)
getattr(st, kind)(text)
st.markdown("**2. Wann entscheidet die Form, und wann die Kopplung?** Bei sicheren Ausgängen nur, wenn Σk von 1 abweicht; die Kopplung ändert die multiplikative Form, die additive nicht (gemessen für zwei Ziele und Zwei-Punkt-Lose).")
kind, text = ev.kern_two(form_row, kopp_row)
getattr(st, kind)(text)
st.markdown("**3. Wie riskant ist es, Mittelwerte einzusetzen?** Der Erwartungsnutzen mit gekrümmtem Einzelnutzen wählt oft anders als die Rechnung mit den Attribut-Erwartungswerten.")
kind, text = ev.kern_three(risk_row, form_risk)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wann entscheidet die Form (sichere Ausgänge)?")
st.caption("Zufallsmatrizen (keine Fälle): n Alternativen, m Attribute, Einzelnutzen linear und gleichverteilt. Additiv mit Gewichten $w_i \\propto 1/(i+1)$, multiplikativ mit $k_i = \\Sigma k \\cdot w_i$. "
           "Gezählt werden Matrizen, in denen Platz 1 verschieden ist. Das Experiment ist die Brücke zu den Demos zum Mehrkriterien-Ranking (Gewichtswechsel, SMAA, Rangumkehr dort bei sicheren Matrizen).")
e1, e2, e3, e4 = st.columns(4)
e1.slider("Σk der multiplikativen Form", *bounds("sumk_form_slider"), step=C.SUMK_STEP, key="sumk_form_slider", help="Σk = 1: die Formen sind gleich. Alles andere: Kreuzterm.")
e2.slider("Attribute m", *bounds("m_slider"), key="m_slider")
e3.slider("Alternativen n", *bounds("n_form_slider"), key="n_form_slider")
e4.slider("Zahl der Matrizen", *bounds("p_form_slider"), step=C.P_STEP, key="p_form_slider")
st.plotly_chart(build_form_bars(form_rows), width="stretch", key="form_bars")
form_df = pd.DataFrame([{"Σk": ev.de(r.sum_k, 1), "K": ev.de(r.K, 3, True), "Platz 1 verschieden": f"{r.changed} von {r.total}", "Anteil (%)": ev.pct(r.changed, r.total),
                         "Gleichstände": r.ties, "Kendall-Tau, Mittel": ev.de(r.tau_mean, 3), "Kendall-Tau, kleinster": ev.de(r.tau_min, 3)} for r in form_rows])
st.dataframe(form_df, hide_index=True, width="stretch")
idx, found, add_top, mult_top = _form_example(P_form, n_form, m_form, seed_i, sum_form)
st.caption(f"Beispielmatrix Nr. {idx + 1} (die erste, in der Platz 1 wechselt)" if found else f"Beispielmatrix Nr. {idx + 1} (in keiner Matrix wechselt Platz 1; gezeigt wird die erste)")
st.dataframe(pd.DataFrame({"Platz": list(range(1, len(add_top) + 1)), "additiv: Alternative": [a + 1 for a, _ in add_top], "additiv: U": [ev.de(v, 4) for _, v in add_top],
                           "multiplikativ: Alternative": [a + 1 for a, _ in mult_top], "multiplikativ: U": [ev.de(v, 4) for _, v in mult_top]}), hide_index=True, width="stretch")
kind, text = ev.verdict_form(form_row, n_form, m_form)
getattr(st, kind)(text)
cont = _continuity(P_form, n_form, m_form, seed_i)
st.caption(ev.text_continuity(cont))

st.markdown("---")

# --- Experiment 3 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Mittelwerte einsetzen, die Falle")
st.markdown("**(a) Zufallsprobleme.** 10 Alternativen, 3 Attribute, S gleich wahrscheinliche Szenarien, Werte gleichverteilt und unabhängig. Verglichen wird der **Erwartungsnutzen** mit gekrümmtem Einzelnutzen "
            "(ρ geteilt durch die Spanne) mit der Rechnung, die die **Attribut-Erwartungswerte** einsetzt (derselben Form, linear).")
r1, r2, r3, r4 = st.columns(4)
r1.select_slider("ρ geteilt durch die Spanne", options=C.RHO_NORM_OPTIONS, key="rho_norm_select", format_func=lambda x: f"{x:g}".replace(".", ","),
                 help="Kleiner heißt stärker gekrümmt. Der Kapazitätsfall liegt bei 555,8/300 = 1,85.")
r2.slider("Szenarien S", *bounds("s_slider"), key="s_slider")
r3.slider("Zahl der Probleme", *bounds("p_risk_slider"), step=C.P_STEP, key="p_risk_slider")
r4.radio("Form", C.FORM_OPTIONS, key="form_risk_radio", help="Multiplikativ: Σk = 0,5 mit w_i ∝ 1/(i+1).")
st.plotly_chart(build_risk_curve(risk_rows), width="stretch", key="risk_curve")
kind, text = ev.verdict_risk(risk_row, S_risk, form_risk)
getattr(st, kind)(text)
with st.spinner("Rechne die Szenarienzahlen (meist unter einer Sekunde) …"):
    sc_rows = _risk_scenarios(P_risk, seed_i, rho_norm, form_risk)
st.dataframe(pd.DataFrame([{"Szenarien S": s, "Platz 1 ändert sich": f"{r.changed} von {r.total}", "Anteil (%)": ev.pct(r.changed, r.total)} for s, r in sc_rows]), hide_index=True, width="stretch")
kind, text = ev.verdict_scenarios(sc_rows)
getattr(st, kind)(text)

st.markdown("**(b) Kapazitätsfall: die Kipp-Risikotoleranz.** Kurven $EU(G) - EU(K)$ über ρ mit den Konstanten und Servicegraden aus Experiment 1. Positiv heißt: G wird gewählt; die Punkte sind die Nullstellen.")
st.plotly_chart(build_rho_curves(hand), width="stretch", key="rho_curves")
kind, text = ev.verdict_kipp(hand)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Gleiche Ränder, andere Kopplung")
st.caption("Zufallsprobleme (keine Fälle): zwei Attribute, je Alternative und Attribut ein Zwei-Punkt-Los (Werte gleichverteilt, je ½), Einzelnutzen linear, $k = (0{,}6\\,\\Sigma k;\\; 0{,}4\\,\\Sigma k)$. "
           "Dieselben Ränder werden dreimal gepaart: gleichläufig (hoch mit hoch), unabhängig, gegenläufig (hoch mit tief).")
k_1, k_2, k_3 = st.columns(3)
k_1.slider("Σk der multiplikativen Form", *bounds("sumk_kopp_slider"), step=C.SUMK_STEP, key="sumk_kopp_slider", help="Σk = 1: auch die multiplikative Form kennt die Kopplung nicht.")
k_2.slider("Alternativen n", *bounds("n_kopp_slider"), key="n_kopp_slider")
k_3.slider("Zahl der Probleme", *bounds("p_kopp_slider"), step=C.P_STEP, key="p_kopp_slider")
st.markdown("| Kopplung | hoch/hoch | hoch/tief | tief/hoch | tief/tief |\n|---|---|---|---|---|\n"
            "| gleichläufig | ½ | 0 | 0 | ½ |\n| unabhängig | ¼ | ¼ | ¼ | ¼ |\n| gegenläufig | 0 | ½ | ½ | 0 |")
st.plotly_chart(build_coupling_bars(kopp_rows), width="stretch", key="coupling_bars")
ex = _coupling_example(P_kopp, n_kopp, seed_i, sum_kopp)
st.dataframe(pd.DataFrame([{"Alternative": e["Alternative"], "Attribut 1 (hoch / tief)": f"{ev.de(e['a_hoch'], 3)} / {ev.de(e['a_tief'], 3)}",
                            "Attribut 2 (hoch / tief)": f"{ev.de(e['b_hoch'], 3)} / {ev.de(e['b_tief'], 3)}", "EU additiv (jede Kopplung)": ev.de(e["additiv"], 4),
                            "EU gleichläufig": ev.de(e["gleichläufig"], 4), "EU unabhängig": ev.de(e["unabhängig"], 4), "EU gegenläufig": ev.de(e["gegenläufig"], 4)} for e in ex]),
             hide_index=True, width="stretch")
kind, text = ev.verdict_coupling(kopp_row, n_kopp)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Ziele sind nutzenunabhängig, die Form gilt** | Hier Annahme, nicht geprüft. Ohne sie ist die Produktform nicht begründet. | – |
| **Die $k_i$ sind bekannt** | Die Erhebung (Lotterie, Rauschen, Verfahrenseffekte) ist hier nicht gemessen; eine falsche Summe Σk macht aus einer additiven Wahl eine multiplikative. | [`nutzen-erheben-demo`](https://github.com/sebastian-hanisch/nutzen-erheben-demo) |
| **Zwei Ziele, Zwei-Punkt-Lose** | Kopplung ist hier nur für zwei Attribute gemessen; mit mehr Attributen ist sie ein Satz von Paarungen, nicht gemessen. | – |
| **Die additive Näherung ist meist unauffällig** | Bei sicheren Ausgängen ändert die Form Platz 1 in diesen Zufallsmatrizen in der Voreinstellung (n = 20, m = 5) in 3 bis 9 % der Matrizen (bei Σk ≠ 1), bei m = 2 und n = 40 in bis zu 29 %; bei Risiko in den Attributen und bei Kopplung ist der Unterschied größer (Experimente 3 und 4). | – |
| **Alles ist ein Modell** | Auszahlungen, Servicegrade und $k_i$ sind ausgedacht, die Zufallsprobleme keine Fälle: die Häufigkeiten gelten für diese Generatoren. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Alternativen, deren Ausgänge zwei Zahlen haben (Gewinn $x_1$ in GE, Servicegrad $x_2$) und eine gemeinsame Verteilung der Ausgänge; gesucht ist die Alternative mit dem größten
Erwartungsnutzen.

| Symbol | Bedeutung |
|---|---|
| $u_i$ | Einzelnutzen des Ziels $i$ in $[0, 1]$ (schlechtester Wert 0, bester 1) |
| $k_i$ | Skalierungskonstante: Nutzen der Ecke „Ziel $i$ am besten, alle anderen am schlechtesten“ |
| $K$, $c$ | $K$ aus $1 + K = \prod_i (1 + K k_i)$; $c = 1 - k_1 - k_2 = K k_1 k_2$ (zwei Ziele) |
| $\rho$ | Risikotoleranz des Gewinns in GE ($\rho = \infty$: linear); $\rho/\text{Spanne}$ im Zufallsexperiment |
| $q$ | Kopplung $P(\text{Nachfrage hoch und Engpass})$, $0 \le q \le \tfrac12$ |

**Einzelnutzen (Gewinn).** Mit $x_{\min} = -100$ und $x_{\max} = 200$ GE:
$$u_1(x) = \frac{1 - e^{-(x - x_{\min})/\rho}}{1 - e^{-(x_{\max} - x_{\min})/\rho}}, \qquad u_1(x) = \frac{x - x_{\min}}{x_{\max} - x_{\min}} \text{ für } \rho = \infty$$

**Multiplikative Form** (Produktform):
$$1 + K\,U = \prod_i \bigl(1 + K\,k_i\,u_i\bigr), \qquad 1 + K = \prod_i \bigl(1 + K\,k_i\bigr)$$

**Zwei Ziele ausmultipliziert:**
$$U = k_1 u_1 + k_2 u_2 + c\,u_1 u_2, \qquad c = 1 - k_1 - k_2 = K\,k_1 k_2$$

**Additive Form:**
$$U_{\mathrm{add}} = \sum_i \frac{k_i}{\sum_j k_j}\,u_i$$

**Erwartungsnutzen** über die Zellen $z$ der gemeinsamen Verteilung:
$$\mathrm{EU} = \sum_z p_z\,U\bigl(u(z)\bigr) = k_1 \bar u_1 + k_2 \bar u_2 + c\,\bigl(\bar u_1 \bar u_2 + \mathrm{Cov}(u_1, u_2)\bigr)$$
Die additive Form hängt nur von $\bar u_1$ und $\bar u_2$ ab, also nur von den Rändern; der Kovarianzterm gehört allein zur multiplikativen Form (und verschwindet bei $c = 0$, also bei $\Sigma k = 1$).

**Zwei Zwei-Punkt-Lose mit gleichen Rändern** (Werte $a_h > a_t$ und $b_h > b_t$, je mit Wahrscheinlichkeit ½):
$$\mathrm{EU}_{\text{gleichläufig}} - \mathrm{EU}_{\text{gegenläufig}} = c\cdot\tfrac12\,(a_h - a_t)(b_h - b_t)$$

**Kapazitätsfall.** $\mathrm{EU}(G)$ ist in $q$ linear (die Ränder sind fest), $\mathrm{EU}(K)$ hängt nicht von $q$ ab; die Kipp-Kopplung folgt aus zwei Stützstellen:
$$q^{*} = \frac{d_0}{d_0 - d_{1/2}}\cdot\tfrac12, \qquad d_q = \mathrm{EU}_q(G) - \mathrm{EU}(K)$$

**Rechenfalle.** Bei $\Sigma k$ nahe 1 ist $K$ nahe 0. Die direkt ausgewertete Gleichung $\bigl(\prod_i (1 + K k_i) - 1 - K\bigr)/K = 0$ löscht sich aus; stabil löst man mit den Koeffizienten der Produktentwicklung
$$(e_1 - 1) + e_2 K + e_3 K^2 + \dots = 0, \qquad e_j = \text{$j$-te elementarsymmetrische Funktion der } k_i.$$
Gelten die Annahmen nicht (Nutzenunabhängigkeit, bekannte $k_i$, nur zwei Ziele), steht das unter 🚧 oben.

**Literatur.** Produktform und Skalierungskonstanten sind Lehrbuchstoff der Mehrattribut-Nutzentheorie (Keeney und Raiffa 1976; im Sammelband Bell, Keeney, Raiffa (Hrsg.) 1977 behandelt Kap. 7
von Fishburn die Bedingungen für additive und multiplikative Formen); die Einleitung des Kapitels ist gelesen, die Bedingungen im Wortlaut nicht. Alle Zahlen hier sind eigene Nachrechnung in diesem Modell.
        """
    )

sync_query_params({
    "k1_slider": float(k1), "k2_slider": float(k2), "seed_input": seed_i, "q_slider": float(q), "rho_select": float(rho), "sg_slider": int(s_g), "sk_slider": int(s_k),
    "sumk_form_slider": sum_form, "m_slider": m_form, "n_form_slider": n_form, "p_form_slider": P_form, "rho_norm_select": rho_norm, "s_slider": S_risk,
    "p_risk_slider": P_risk, "form_risk_radio": form_risk, "sumk_kopp_slider": sum_kopp, "n_kopp_slider": n_kopp, "p_kopp_slider": P_kopp,
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html)."
)
