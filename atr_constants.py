"""Konstanten: Regler-Grenzen, feste Annahmen des Kapazitätsfalls, Farben und Presets.

Der Kapazitätsfall ist ein **ausgedachtes Zahlenbeispiel** (Geldeinheiten GE, Servicegrade in %): ein Modell, kein Fall und keine Messung an Menschen."""

import math

# --- Kapazitätsfall (ausgedacht) -------------------------------------------------------------------------------------------------------------
GAIN_LO, GAIN_HI = -100, 200            # Spanne des Gewinns in GE (schlechtester und bester Wert aller Alternativen): 300 GE
GAIN_SPAN = GAIN_HI - GAIN_LO
GAIN_G_HIGH, GAIN_G_LOW = 200, -100     # Alternative G (groß bauen): Gewinn bei hoher bzw. niedriger Nachfrage
GAIN_K = 30                             # Alternative K (klein bauen oder mieten): sicherer Gewinn
P_DEMAND_HIGH = 0.5                     # Nachfrage hoch und Lieferlage normal: je 1/2 (Ränder fest, nur die Kopplung q ändert sich)
RHO_STAR_SINGLE = 555.801431            # Kipp-Risikotoleranz für den Gewinn allein (G gegen K), GE; Test: tests/test_oracle_attribute_risiko.py
RHO_GRID = (5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 400.0, 555.8, 700.0, 1000.0, 2000.0, 5000.0, 1e4, 1e5, 1e6, 1e9)  # Raster für die Kipp-Suche

# --- Regler ----------------------------------------------------------------------------------------------------------------------------------
K_MIN, K_MAX, K_STEP = 0.05, 0.95, 0.05
DEFAULT_K1, DEFAULT_K2 = 0.40, 0.30
Q_MIN, Q_MAX, Q_STEP, DEFAULT_Q = 0.0, 0.5, 0.05, 0.25
RHO_OPTIONS = (math.inf, 5000.0, 2000.0, 1000.0, 555.8, 400.0, 300.0, 200.0, 100.0)   # ρ des Gewinns in GE (∞ = linear)
DEFAULT_RHO = math.inf
SERVICE_MIN, SERVICE_MAX, SERVICE_STEP = 0, 100, 5
DEFAULT_SERVICE_G, DEFAULT_SERVICE_K = 50, 60                                         # Servicegrad im Engpass in %

SUMK_MIN, SUMK_MAX, SUMK_STEP = 0.5, 1.5, 0.1
DEFAULT_SUMK_FORM, DEFAULT_SUMK_KOPPLUNG = 0.5, 0.5
M_MIN, M_MAX, DEFAULT_M = 2, 8, 5
N_MIN, N_MAX, DEFAULT_N_FORM = 3, 40, 20
DEFAULT_N_KOPPLUNG = 10
P_MIN, P_MAX, P_STEP, DEFAULT_P = 1000, 5000, 1000, 5000
RHO_NORM_OPTIONS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1)                                    # ρ geteilt durch die Spanne des Attributs
DEFAULT_RHO_NORM = 2.0
S_MIN, S_MAX, DEFAULT_S = 2, 10, 2
FORM_ADD = "additiv"
FORM_MULT = "multiplikativ (Σk = 0,5)"
FORM_OPTIONS = (FORM_ADD, FORM_MULT)
RISK_SUM_K = 0.5
RISK_M = 3                                                                            # Zahl der Attribute im Risikoexperiment
SEED_MAX, DEFAULT_SEED = 999999, 1017

TIE = 1e-12                  # Abstand zweier Spitzenwerte, unter dem ein Gleichstand gezählt wird (kein Wechsel)
SUM_K_VALUES = (0.5, 0.8, 1.0, 1.2, 1.5)  # Σk-Werte der Vergleichstabellen

COLOR_G = "#2F6B65"
COLOR_K = "#c4742b"
COLOR_ADD = "#4c78a8"
COLOR_MULT = "#b5483b"
COLOR_CROSS = "rgba(181,72,59,0.45)"
COLOR_GRID = "rgba(150,150,150,0.35)"

# --- Presets ---------------------------------------------------------------------------------------------------------------------------------
# Schlüssel = Schlüssel der Regler in atr_presets.SETTING_SPECS (ohne „_slider“ usw.); fehlende Schlüssel meinen den Standardwert.
PRESETS = {
    "Von Hand (Brüche, ρ = ∞)": {},
    "Σk = 1 (additiv)": {"k1": 0.40, "k2": 0.60, "sumk_form": 1.0, "sumk_kopplung": 1.0},
    "Gegenläufig (q = ½)": {"q": 0.5},
    "Risikoscheu (ρ = 555,8 GE)": {"rho": 555.8, "q": 0.0},
    "Gleiche Ränder, andere Kopplung": {"sumk_kopplung": 1.5},
    "Zufallsprobleme, Σk = 0,5": {"sumk_form": 0.5, "sumk_kopplung": 0.5, "rho_norm": 0.5, "form_risk": FORM_MULT},
}
PRESET_HELP = {
    "Von Hand (Brüche, ρ = ∞)": "Kapazitätsfall mit k = (2/5, 3/10), unabhängiger Kopplung (q = ¼) und linearem Gewinn: alles in Brüchen, von Hand nachrechenbar.",
    "Σk = 1 (additiv)": "k = (0,4; 0,6): die Summe ist 1, K = 0, die multiplikative Form ist die additive. Kein Formunterschied, auch in den Zufallsproblemen nicht.",
    "Gegenläufig (q = ½)": "Hohe Nachfrage kommt immer mit Engpass (q = ½): die multiplikative Form wählt jetzt K statt G, die additive weiter G.",
    "Risikoscheu (ρ = 555,8 GE)": "Risikotoleranz des Gewinns 555,8 GE, hohe Nachfrage und Engpass treten nie zusammen auf (q = 0): die beiden Formen wählen verschieden.",
    "Gleiche Ränder, andere Kopplung": "Σk = 1,5 im Kopplungsexperiment: dieselben Ränder, nur anders gepaart, ändern Platz 1 in mehr als 10 % der Probleme (multiplikativ, gleichläufig gegen gegenläufig).",
    "Zufallsprobleme, Σk = 0,5": "Σk = 0,5 in beiden Zufallsexperimenten, stark gekrümmter Einzelnutzen (ρ/Spanne = 0,5), multiplikative Form im Risikoexperiment.",
}
