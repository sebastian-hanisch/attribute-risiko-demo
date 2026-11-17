"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; je Preset die erwartete Verdikt-Zeile (kein Preset ohne Wirkung meldet eine Wirkung)."""

import math
from fractions import Fraction as Fr

import pytest

import atr_constants as C
import atr_evaluation as ev
import atr_presets as P
import atr_sweeps as SW
import atr_utility as U


def test_every_preset_has_help_and_only_known_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) <= set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            spec = P.SETTING_SPECS[P.PRESET_KEYS[key]]
            if spec.options is not None:
                assert value in spec.options, (name, key)
            else:
                assert spec.lo <= value <= spec.hi, (name, key)


def test_default_preset_equals_the_default_settings_and_presets_are_distinct():
    assert C.PRESETS["Von Hand (Brüche, ρ = ∞)"] == {}
    resolved = [tuple(sorted((k, v) for k, v in p.items())) for p in C.PRESETS.values()]
    assert len(set(resolved)) == len(resolved)
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_clamp_snaps_to_options_and_limits():
    assert P.clamp(P.SETTING_SPECS["rho_select"], 560.0) == 555.8
    assert P.clamp(P.SETTING_SPECS["rho_select"], math.inf) == math.inf
    assert P.clamp(P.SETTING_SPECS["rho_norm_select"], 1.1) == 1.0
    assert P.clamp(P.SETTING_SPECS["form_risk_radio"], "unsinn") == C.FORM_ADD
    assert P.clamp(P.SETTING_SPECS["q_slider"], 0.9) == 0.5 and P.clamp(P.SETTING_SPECS["k1_slider"], -3.0) == C.K_MIN
    assert P.same(0.30000000000000004, 0.3) and not P.same(0.3, 0.35) and P.same(math.inf, math.inf) and not P.same(math.inf, 5.0)


def _hand(preset):
    g = lambda key, default: preset.get(key, default)  # noqa: E731
    return ev.HandSettings(U.as_fraction(g("k1", C.DEFAULT_K1)), U.as_fraction(g("k2", C.DEFAULT_K2)), U.as_fraction(g("q", C.DEFAULT_Q)), g("rho", C.DEFAULT_RHO),
                           Fr(g("s_g", C.DEFAULT_SERVICE_G), 100), Fr(g("s_k", C.DEFAULT_SERVICE_K), 100))


def test_preset_von_hand_meldet_die_handwerte():
    kind, text = ev.verdict_hand(ev.analyse_hand(_hand(C.PRESETS["Von Hand (Brüche, ρ = ∞)"])))
    assert kind == "info"
    assert "43/80" in text and "194/375" in text and "17/28" in text and "62/105" in text and "q* = 173/450" in text and "wählt G" in text


def test_preset_summe_eins_meldet_keinen_formunterschied():
    preset = C.PRESETS["Σk = 1 (additiv)"]
    kind, text = ev.verdict_hand(ev.analyse_hand(_hand(preset)))
    assert kind == "success" and "die multiplikative Form ist die additive" in text and "Die Form entscheidet" not in text
    form = SW.form_table(C.DEFAULT_P, C.DEFAULT_N_FORM, C.DEFAULT_M, C.DEFAULT_SEED, (preset["sumk_form"],))[0]
    assert ev.verdict_form(form, C.DEFAULT_N_FORM, C.DEFAULT_M)[0] == "success" and form.changed == 0
    kopp = SW.coupling_table(C.DEFAULT_P, C.DEFAULT_N_KOPPLUNG, C.DEFAULT_SEED, (preset["sumk_kopplung"],))[0]
    assert ev.verdict_coupling(kopp, C.DEFAULT_N_KOPPLUNG)[0] == "success" and kopp.same_vs_opposite[0] == 0


def test_preset_gegenlaeufig_laesst_die_formen_verschieden_waehlen():
    kind, text = ev.verdict_hand(ev.analyse_hand(_hand(C.PRESETS["Gegenläufig (q = ½)"])))
    assert kind == "warning" and "Die Form entscheidet" in text and "wählt K" in text and "wählt G" in text


def test_preset_risikoscheu_laesst_die_formen_verschieden_waehlen():
    r = ev.analyse_hand(_hand(C.PRESETS["Risikoscheu (ρ = 555,8 GE)"]))
    kind, text = ev.verdict_hand(r)
    assert kind == "warning" and "Die Form entscheidet" in text
    assert ev.winner(r.d_mult) == "G" and ev.winner(r.d_add) == "K"


def test_preset_gleiche_raender_andere_kopplung_wechselt_platz_eins_in_mehr_als_zehn_prozent():
    sk = C.PRESETS["Gleiche Ränder, andere Kopplung"]["sumk_kopplung"]
    kopp = SW.coupling_table(C.DEFAULT_P, C.DEFAULT_N_KOPPLUNG, C.DEFAULT_SEED, (sk,))[0]
    assert kopp.same_vs_opposite[0] > 0.10 * kopp.total and kopp.add_same_vs_opposite[0] == 0
    assert "Σk = 1,5" in C.PRESET_HELP["Gleiche Ränder, andere Kopplung"] and "mehr als 10 %" in C.PRESET_HELP["Gleiche Ränder, andere Kopplung"]


def test_preset_zufallsprobleme_summe_halb_nennt_die_gezaehlten_probleme():
    p = C.PRESETS["Zufallsprobleme, Σk = 0,5"]
    form = SW.form_table(C.DEFAULT_P, C.DEFAULT_N_FORM, C.DEFAULT_M, C.DEFAULT_SEED, (p["sumk_form"],))[0]
    _, text = ev.verdict_form(form, C.DEFAULT_N_FORM, C.DEFAULT_M)
    assert f"{form.changed} von {form.total} Matrizen" in text and form.changed > 0
    row = SW.risk_table(C.DEFAULT_P, C.DEFAULT_S, C.DEFAULT_SEED, p["form_risk"], (p["rho_norm"],))[0]
    assert f"{row.changed} von {row.total}" in ev.verdict_risk(row, C.DEFAULT_S, p["form_risk"])[1]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_hilfetext_passt_zur_messung_der_kerngroesse(name):
    """Die Hilfetexte nennen nur Zahlen, die in den Presets stehen: k, q, ρ, Σk."""
    text = C.PRESET_HELP[name]
    for key, value in C.PRESETS[name].items():
        if key == "q" and value == 0.5:
            assert "q = ½" in text
        if key == "sumk_form" and value == 1.0:
            assert "Σk = 1" in text or "Summe ist 1" in text
