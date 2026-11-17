"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier."""

import math
from fractions import Fraction as Fr

import atr_constants as C
import atr_evaluation as ev
import atr_sweeps as SW


def test_formatierung_dezimalkomma_und_bruch():
    assert ev.de(-0.5, 2) == "−0,50" and ev.de(0.5, 1, True) == "+0,5"
    assert ev.fr(Fr(23, 40)) == "23/40 (0,5750)" and ev.fr(Fr(1)) == "1" and ev.fr(0.5) == "0,5000" and ev.fr(Fr(-1, 300), 4) == "−1/300 (−0,0033)"
    assert ev.pct(448, 5000) == "9,0" and ev.pct(0, 0) == "0,0"
    assert ev.winner(0.1) == "G" and ev.winner(-0.1) == "K" and ev.winner(1e-13) == "keine (Gleichstand)"


def _row(sum_k, K, changed, ties=0, total=5000, tau=0.95, tau_min=0.84):
    return SW.FormRow(sum_k, K, changed, ties, total, tau, tau_min)


def test_verdict_form_summe_eins_meldet_null_und_success():
    kind, text = ev.verdict_form(_row(1.0, 0.0, 0), 20, 5)
    assert kind == "success" and "5000 Matrizen, n = 20 Alternativen, m = 5 Attribute" in text
    assert "Platz 1 ist in 0 von 5000 Matrizen verschieden (0 Gleichstände)" in text and "die multiplikative Form ist die additive" in text


def test_verdict_form_nennt_zaehlung_anteil_standardfehler_und_tau():
    kind, text = ev.verdict_form(_row(0.5, 3.72, 450, tau=0.9504, tau_min=0.8421), 20, 5)       # p = 0,09: Standardfehler √(0,09·0,91/5000) = 0,40 Punkte
    assert kind == "info"
    assert "Σk = 0,5 (K = +3,72)" in text and "450 von 5000 Matrizen verschieden (9,0 %, Standardfehler 0,4 Punkte)" in text
    assert "in 4550 Matrizen ist die additive Näherung für Platz 1 unauffällig" in text
    assert "im Mittel 0,950, kleinster Wert 0,842 (erste 1000 Matrizen)" in text


def test_verdict_form_ties_werden_nicht_als_unauffaellig_gezaehlt():
    _, text = ev.verdict_form(_row(1.5, -0.7, 100, ties=20, total=1000), 10, 3)
    assert "in 880 Matrizen ist die additive Näherung" in text and "erste 1000 Matrizen" in text


def test_verdict_risk_nennt_anteil_und_kapazitaetsfall():
    kind, text = ev.verdict_risk(SW.RiskRow(2.0, 250, 0, 5000), 2, C.FORM_ADD)                   # 5,0 %, Standardfehler 0,31
    assert kind == "info" and "ρ/Spanne = 2,00" in text and "250 von 5000 Problemen (5,0 %, Standardfehler 0,3 Punkte)" in text and "ρ/Spanne = 1,85" in text
    assert "Kopplung der Szenarien" not in text
    assert "Kopplung der Szenarien" in ev.verdict_risk(SW.RiskRow(2.0, 250, 0, 5000), 2, C.FORM_MULT)[1]


def test_verdict_scenarios_nicht_sinkend_und_sinkend():
    rows = [(2, SW.RiskRow(0.5, 900, 0, 3000)), (10, SW.RiskRow(0.5, 1000, 0, 3000))]
    text = ev.verdict_scenarios(rows)[1]
    assert "S = 2: 30,0 %, S = 10: 33,3 %" in text and "sinkt der Anteil in diesen Zufallsproblemen nicht" in text
    rows = [(2, SW.RiskRow(0.5, 900, 0, 3000)), (10, SW.RiskRow(0.5, 600, 0, 3000))]
    assert "sinkt der Anteil in diesen Zufallsproblemen doch" in ev.verdict_scenarios(rows)[1]


def _coupling(sum_k, c, same_opp, add=0, dmin=0.0, dmax=0.2):
    return SW.CouplingResult(sum_k, c, 5000, (same_opp, 0), (300, 0), (250, 0), (600, 0), (add, 0), 0.02, dmin, dmax)


def test_verdict_coupling_summe_eins_und_vorzeichen():
    kind, text = ev.verdict_coupling(_coupling(1.0, 0.0, 0), 10)
    assert kind == "success" and "Additiv ändert sich Platz 1 mit der Kopplung in 0 von 5000 Problemen" in text and "auch multiplikativ in 0 von 5000" in text
    kind, text = ev.verdict_coupling(_coupling(0.5, 0.5, 570, dmin=0.0, dmax=0.2254), 10)
    assert kind == "info" and "gleichläufig gegen gegenläufig wechselt Platz 1 in 570 von 5000 Problemen (11,4 %)" in text
    assert "gleichläufig gegen unabhängig in 300" in text and "gegenläufig gegen unabhängig in 250" in text
    assert "zwischen +0,000 und +0,225" in text and "gleichläufig vor gegenläufig" in text
    _, text = ev.verdict_coupling(_coupling(1.5, -0.5, 650, dmin=-0.2254, dmax=0.0), 10)
    assert "c = −0,5" in text and "gegenläufig vor gleichläufig" in text
    _, text = ev.verdict_coupling(_coupling(0.5, 0.5, 570, dmin=-0.01, dmax=0.2), 10)             # Vorzeichen verletzt: kein Satz darüber
    assert "Vorzeichen von c entscheidet" not in text


def test_verdict_hand_summe_eins_fehlende_kipp_kopplung_und_float():
    hs = ev.HandSettings(Fr(2, 5), Fr(3, 5), Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5))
    kind, text = ev.verdict_hand(ev.analyse_hand(hs))
    assert kind == "success" and "Σk = 1, also c = 0 und K = 0" in text and "Beide wählen K" in text and "−1/300 (−0,0033)" in text
    hs = ev.HandSettings(Fr(2, 5), Fr(3, 10), Fr(1, 4), 555.8, Fr(1, 2), Fr(3, 5))
    kind, text = ev.verdict_hand(ev.analyse_hand(hs))
    assert "ρ = 555,8 GE" in text and "q* = " in text


def test_verdict_kipp_nennt_alle_fuenf_kippstellen():
    hs = ev.HandSettings(Fr(2, 5), Fr(3, 10), Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5))
    text = ev.verdict_kipp(ev.analyse_hand(hs))[1]
    assert "ρ* = 555,8 GE" in text and "additiv bei 1268,0 GE" in text and "multiplikativ bei 410,9 GE (q = 0)" in text and "1174,0 GE (q = ¼)" in text
    assert "q = ½: kein Wechsel im Raster" in text and "Servicegrad im Engpass G 50 %, K 60 %" in text and "nicht „für jedes ρ“" in text


def test_kernfragen_sind_kurz_und_nennen_die_zahlen():
    r = ev.analyse_hand(ev.HandSettings(Fr(2, 5), Fr(3, 10), Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5)))
    assert "Σk = 7/10: c = 1 − Σk = 3/10 ist positiv, der Kreuzterm belohnt Ausgewogenheit; K = 5/2" in ev.kern_one(r)[1]
    r1 = ev.analyse_hand(ev.HandSettings(Fr(2, 5), Fr(3, 5), Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5)))
    assert "c = 0" in ev.kern_one(r1)[1]
    assert "Platz 1 verschieden in 450 von 5000 Matrizen" in ev.kern_two(_row(0.5, 3.7, 450), _coupling(0.5, 0.5, 570))[1]
    assert "gleichläufig gegen gegenläufig in 570 von 5000 Problemen, additiv in 0" in ev.kern_two(_row(0.5, 3.7, 450), _coupling(0.5, 0.5, 570))[1]
    assert "250 von 5000 Problemen (5,0 %)" in ev.kern_three(SW.RiskRow(2.0, 250, 0, 5000), C.FORM_ADD)[1]


def test_text_continuity_beide_zweige():
    cont = {"K_stable": 2.61e-6, "K_naive": 1.00005e-12, "K_series": 2.61e-6, "changed_stable": 0, "changed_naive": 2, "total": 5000}
    text = ev.text_continuity(cont)
    assert "2,610e-06" in text and "1,000e-12" in text and "scheinbar 2 von 5000" in text and "0 von 5000 Matrizen mit anderem Platz 1" in text
    cont["changed_naive"] = None
    assert "kein Vorzeichenintervall" in ev.text_continuity(cont)
