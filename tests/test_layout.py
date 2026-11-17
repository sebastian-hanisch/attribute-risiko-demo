"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten stellt sicher, dass wirklich JEDER Regler
(Seitenleiste und Hauptbereich beim Experiment) dort steht: sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import atr_constants as C
import atr_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "select_slider", "number_input", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for bereich in (at.sidebar, at.main):
        for art in REGLER_ARTEN:
            try:
                return getattr(bereich, art)(key=key)
            except KeyError:
                continue
    raise AssertionError(f"Regler {key} nicht gefunden")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for bereich in (at.sidebar, at.main) for art in REGLER_ARTEN for w in getattr(bereich, art) if w.key}
    assert schluessel == set(P.SETTING_SPECS), f"Regler ohne Eintrag: {sorted(schluessel - set(P.SETTING_SPECS))}, Einträge ohne Regler: {sorted(set(P.SETTING_SPECS) - schluessel)}"


def _passt_zu_preset(state_key, wert) -> bool:
    """Würde ein Preset genau den Standardwerten mit diesem einen geänderten Regler entsprechen? (Dann ist die Hervorhebung zu Recht da.)"""
    key_of = {sk: k for k, sk in P.PRESET_KEYS.items()}
    for preset in C.PRESETS.values():
        if all(P.same(wert if sk == state_key else spec.default, preset.get(key_of[sk], spec.default)) for sk, spec in P.SETTING_SPECS.items()):
            return True
    return False


def _anderer_wert(state_key, spec):
    """Ein Wert ungleich dem Standard, der nicht zufällig selbst ein Preset ergibt."""
    kandidaten = list(spec.options) if spec.options is not None else [spec.hi, spec.lo, spec.lo + (spec.hi - spec.lo) / 4]
    return next(w for w in kandidaten if not P.same(w, spec.default) and not _passt_zu_preset(state_key, w))


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    regler = _regler(at, state_key)
    regler.set_value(_anderer_wert(state_key, P.SETTING_SPECS[state_key])).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    name = next(iter(C.PRESETS))
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_permalink_uebernimmt_werte_und_begrenzt_sie():
    at = AppTest.from_file(APP, default_timeout=180)
    at.query_params["q"] = "0.5"
    at.query_params["k1"] = "9.0"            # über der Grenze: wird begrenzt
    at.query_params["rho"] = "inf"
    at.query_params["rn"] = "0.3"            # kein Auswahlwert: der nächste (0,25)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.session_state["q_slider"] == 0.5 and at.session_state["k1_slider"] == C.K_MAX
    assert at.session_state["rho_select"] == float("inf") and at.session_state["rho_norm_select"] == 0.25
