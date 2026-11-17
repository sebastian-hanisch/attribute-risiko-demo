"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS (auch die im Hauptbereich beim Experiment): ein Preset ist aktiv, wenn jeder Regler dem Preset entspricht."""

import math
import random
from dataclasses import dataclass

import streamlit as st

import atr_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    options: tuple | None = None   # Auswahl-Regler: erlaubte Werte (Zahlen: der nächste erlaubte Wert wird genommen)


SETTING_SPECS = {
    "k1_slider": SettingSpec("k1", float, C.DEFAULT_K1, C.K_MIN, C.K_MAX),
    "k2_slider": SettingSpec("k2", float, C.DEFAULT_K2, C.K_MIN, C.K_MAX),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "q_slider": SettingSpec("q", float, C.DEFAULT_Q, C.Q_MIN, C.Q_MAX),
    "rho_select": SettingSpec("rho", float, C.DEFAULT_RHO, options=C.RHO_OPTIONS),
    "sg_slider": SettingSpec("sg", int, C.DEFAULT_SERVICE_G, C.SERVICE_MIN, C.SERVICE_MAX),
    "sk_slider": SettingSpec("sk", int, C.DEFAULT_SERVICE_K, C.SERVICE_MIN, C.SERVICE_MAX),
    "sumk_form_slider": SettingSpec("sf", float, C.DEFAULT_SUMK_FORM, C.SUMK_MIN, C.SUMK_MAX),
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX),
    "n_form_slider": SettingSpec("nf", int, C.DEFAULT_N_FORM, C.N_MIN, C.N_MAX),
    "p_form_slider": SettingSpec("pf", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "rho_norm_select": SettingSpec("rn", float, C.DEFAULT_RHO_NORM, options=C.RHO_NORM_OPTIONS),
    "s_slider": SettingSpec("s", int, C.DEFAULT_S, C.S_MIN, C.S_MAX),
    "p_risk_slider": SettingSpec("pr", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
    "form_risk_radio": SettingSpec("fr", str, C.FORM_ADD, options=C.FORM_OPTIONS),
    "sumk_kopp_slider": SettingSpec("sc", float, C.DEFAULT_SUMK_KOPPLUNG, C.SUMK_MIN, C.SUMK_MAX),
    "n_kopp_slider": SettingSpec("nc", int, C.DEFAULT_N_KOPPLUNG, C.N_MIN, C.N_MAX),
    "p_kopp_slider": SettingSpec("pc", int, C.DEFAULT_P, C.P_MIN, C.P_MAX),
}
PRESET_KEYS = {
    "k1": "k1_slider", "k2": "k2_slider", "seed": "seed_input", "q": "q_slider", "rho": "rho_select", "s_g": "sg_slider", "s_k": "sk_slider",
    "sumk_form": "sumk_form_slider", "m": "m_slider", "n_form": "n_form_slider", "p_form": "p_form_slider", "rho_norm": "rho_norm_select", "s": "s_slider",
    "p_risk": "p_risk_slider", "form_risk": "form_risk_radio", "sumk_kopplung": "sumk_kopp_slider", "n_kopp": "n_kopp_slider", "p_kopp": "p_kopp_slider",
}


def same(a, b) -> bool:
    """Gleichheit mit Toleranz für Gleitkommazahlen (Regler liefern 0,30000000000000004 statt 0,3)."""
    if isinstance(a, float) or isinstance(b, float):
        if isinstance(a, str) or isinstance(b, str):
            return False
        if math.isinf(a) or math.isinf(b):
            return a == b
        return abs(a - b) < 1e-9
    return a == b


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp(spec: SettingSpec, value):
    """Wert auf die Grenzen bzw. die erlaubten Auswahlwerte begrenzen; Gleitkommazahlen auf 4 Stellen gerundet."""
    if spec.options is not None:
        if spec.caster is str:
            return value if value in spec.options else spec.default
        return min(spec.options, key=lambda o: abs(o - value) if not (math.isinf(o) or math.isinf(value)) else (0.0 if o == value else math.inf))
    if spec.caster is float:
        value = round(float(value), 4)
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            if isinstance(value, float) and math.isnan(value):
                continue
            st.session_state[state_key] = clamp(spec, value)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        spec = SETTING_SPECS[state_key]
        st.session_state[state_key] = C.PRESETS[name].get(key, spec.default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(same(st.session_state.get(state_key, spec.default), preset.get(preset_key_of[state_key], spec.default)) for state_key, spec in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
