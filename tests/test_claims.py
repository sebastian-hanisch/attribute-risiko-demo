"""Claims-Tests: jede Zahl der Befunde B1 bis B9 der README mit ihren Voraussetzungen (DEMO-PLAYBOOK §4, §11a).

Voraussetzung: Kapazitätsfall (ausgedacht) mit k = (2/5; 3/10), Servicegrad im Engpass G 1/2 und K 3/5; Zufallsprobleme mit w_i ∝ 1/(i+1), Zufalls-Seed 1017 (Band über die
Zufalls-Seeds 1017, 1, 2, 3, 4), 5000 Matrizen bzw. Probleme, sofern nicht anders genannt.
Plattform: Brüche exakt; Zählungen aus Neu-Rechnung als Band (± 2: expm1, log1p und die Summenreihenfolge können unter Linux in der letzten Binärstelle abweichen); Kipp-Werte
relativ 1e-6. Jede README-Zahl wird hier gerechnet **und** im README-Text gesucht (`_readme_has`)."""

import math
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
import pytest

import atr_constants as C
import atr_evaluation as ev
import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U
from atr_evaluation import de

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8").replace("\\|", "|").replace("\\*", "*")
SEED = C.DEFAULT_SEED
SEEDS = (SEED, 1, 2, 3, 4)
BAND = 2


def _readme_has(*items):
    for n in items:
        assert n in README, n


def _band(counts, expected):
    """Zählung gleich der README-Zahl, Band ± 2."""
    assert len(counts) == len(expected)
    for got, want in zip(counts, expected):
        assert abs(got - want) <= BAND, (got, want)


def _hand(q=Fr(1, 4), rho=math.inf, k1=Fr(2, 5), k2=Fr(3, 10), sg=Fr(1, 2), sk=Fr(3, 5)):
    return ev.analyse_hand(ev.HandSettings(k1, k2, q, rho, sg, sk))


# --- B1 --------------------------------------------------------------------------------------------------------------------------------

def test_b1_handbeispiel():
    k1, k2 = Fr(2, 5), Fr(3, 10)
    K = (1 - k1 - k2) / (k1 * k2)
    assert K == Fr(5, 2) and (1 + K * k1, 1 + K * k2) == (2, Fr(7, 4)) and 2 * Fr(7, 4) == Fr(7, 2) == 1 + K
    assert U.cross_coefficient(k1, k2) == Fr(3, 10)
    assert (U.u2_mult(Fr(1), Fr(1), k1, k2), U.u2_mult(Fr(0), Fr(0), k1, k2), U.u2_mult(Fr(1), Fr(0), k1, k2), U.u2_mult(Fr(0), Fr(1), k1, k2)) == (1, 0, Fr(2, 5), Fr(3, 10))
    half = Fr(1, 2)
    assert U.u2_mult(half, half, k1, k2) == Fr(17, 40) and k1 * half + k2 * half == Fr(7, 20)
    a, b = (Fr(1), Fr(0)), (half, half)
    assert (U.u2_add(*a, k1, k2), U.u2_add(*b, k1, k2)) == (Fr(4, 7), half) and (U.u2_mult(*a, k1, k2), U.u2_mult(*b, k1, k2)) == (Fr(2, 5), Fr(17, 40))
    ka, kb = Fr(7, 10), Fr(3, 5)
    b2 = (Fr(3, 5), Fr(3, 5))
    assert (1 - ka - kb) / (ka * kb) == Fr(-5, 7)
    assert (U.u2_add(*b2, ka, kb), U.u2_add(*a, ka, kb)) == (Fr(3, 5), Fr(7, 13))
    assert (U.u2_mult(*a, ka, kb), U.u2_mult(*b2, ka, kb)) == (Fr(7, 10), Fr(84, 125))
    _readme_has("**5/2**", "2 · 7/4 = 7/2", "$c = 3/10$", "**17/40**", "7/20", "4/7 gegen 1/2", "2/5 gegen 17/40", "$K = -5/7$", "7/13 gegen 3/5", "7/10 gegen 84/125")


# --- B2 --------------------------------------------------------------------------------------------------------------------------------

def test_b2_summe_eins_und_drei_ziele():
    n = 0
    for a in (Fr(1, 5), Fr(2, 5), Fr(1, 2), Fr(9, 10)):
        for u1 in (Fr(0), Fr(1, 3), Fr(1, 2), Fr(1)):
            for u2 in (Fr(0), Fr(1, 3), Fr(1, 2), Fr(1)):
                assert U.u2_mult(u1, u2, a, 1 - a) == a * u1 + (1 - a) * u2
                n += 1
    assert n == 4 * 16
    k = np.array([0.3, 0.25, 0.2])
    roots = [r.real for r in np.roots([k[0] * k[1] * k[2], k[0] * k[1] + k[0] * k[2] + k[1] * k[2], k.sum() - 1.0]) if abs(r.imag) < 1e-12 and r.real > -1]
    assert U.mult_constant(k) == pytest.approx(roots[0], abs=1e-9) and de(U.mult_constant(k), 4) == "1,2289"
    _readme_has("4 Werte von $k_1$, je 16 Wertepaare", "$K = 1{,}2289$")


# --- B3 --------------------------------------------------------------------------------------------------------------------------------

def test_b3_kapazitaetsfall_von_hand():
    r = _hand()
    assert SC.u1_of(30, math.inf) == Fr(13, 30)
    assert r.eu[("K", "mult")] == Fr(194, 375) and r.eu[("G", "add")] == Fr(17, 28) and r.eu[("K", "add")] == Fr(62, 105) and r.d_add == Fr(1, 60)
    for q in (Fr(0), Fr(1, 8), Fr(1, 4), Fr(1, 2)):
        rq = _hand(q)
        assert rq.eu[("G", "mult")] == Fr(23, 40) - Fr(3, 20) * q and rq.eu[("K", "mult")] == Fr(194, 375) and rq.d_add == Fr(1, 60)
    assert [de(float(_hand(q).d_mult), 4, True) for q in (Fr(0), Fr(1, 4), Fr(1, 2))] == ["+0,0577", "+0,0202", "−0,0173"]
    assert r.q_star == Fr(173, 450) and de(float(r.q_star), 4) == "0,3844"
    assert SC.moments(SC.cells("G", Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5)))[2] == 0 and SC.moments(SC.cells("G", Fr(0), math.inf, Fr(1, 2), Fr(3, 5)))[2] == Fr(1, 8)
    _readme_has("$u_1(30) = 13/30$", "**194/375**", "**23/40 − 3q/20**", "17/28", "62/105", "**G − K = 1/60 für jedes $q$**", "+0,0577 / +0,0202 / −0,0173", "**$q^* = 173/450$ (0,3844)**",
                "$\\tfrac18 - \\tfrac{q}{2}$")


# --- B4 --------------------------------------------------------------------------------------------------------------------------------

def test_b4_kipp_risikotoleranz():
    from scipy import optimize
    assert SC.rho_star_single() == pytest.approx(555.801, abs=1e-3) and de(SC.rho_star_single(), 1) == "555,8"
    assert optimize.brentq(lambda r: SC.ce_gain_g(r) - 30.0, 50.0, 1e4, xtol=1e-12) == pytest.approx(SC.rho_star_single(), abs=1e-6)
    r = _hand(k1=Fr(2, 5), k2=Fr(3, 10))
    k = r.kipp
    assert len(k["add"]) == 1 and k["add"][0] == pytest.approx(1268.0, abs=0.05) and de(k["add"][0], 1) == "1268,0"
    assert len(k["mult0"]) == 1 and de(k["mult0"][0], 1) == "410,9" and len(k["mult25"]) == 1 and de(k["mult25"][0], 1) == "1174,0" and k["mult50"] == []
    assert len(C.RHO_GRID) == 17 and C.RHO_GRID[0] == 5.0 and C.RHO_GRID[-1] == 1e9
    # Bei ρ = 555,8: G − K, und die Kipp-Kopplung
    outs = [ev.analyse_hand(ev.HandSettings(Fr(2, 5), Fr(3, 10), U.as_fraction(q), 555.8, Fr(1, 2), Fr(3, 5))) for q in (0.0, 0.25, 0.5)]
    assert [de(float(o.d_add), 4, True) for o in outs] == ["−0,0214"] * 3 and [de(float(o.d_mult), 4, True) for o in outs] == ["+0,0150", "−0,0225", "−0,0600"]
    assert de(float(outs[0].q_star), 4) == "0,1000"
    out200 = ev.analyse_hand(ev.HandSettings(Fr(2, 5), Fr(3, 10), Fr(0), 200.0, Fr(1, 2), Fr(3, 5)))
    assert out200.q_star is None
    assert de(k["mult0"][0] / 555.8, 2) == "0,74" and de(k["add"][0] / 555.8, 2) == "2,28"
    _readme_has("**555,8 GE**", "**additiv 1268,0 GE**", "**multiplikativ 410,9 GE ($q = 0$), 1174,0 GE ($q = \\tfrac14$), kein Wechsel im Raster ($q = \\tfrac12$)**", "−0,0214", "+0,0150 / −0,0225 / −0,0600",
                "$q^* = 0{,}1000$", "bei $\\rho = 200$ GE gibt es keine", "0,74 bis 2,28")


# --- B5 --------------------------------------------------------------------------------------------------------------------------------

def test_b5_form_sichere_ausgaenge():
    rows = SW.form_table(5000, 20, 5, SEED)
    assert [r.sum_k for r in rows] == [0.5, 0.8, 1.0, 1.2, 1.5]
    _band([r.changed for r in rows], [448, 177, 0, 166, 451])
    assert rows[2].changed == 0 and rows[2].ties == 0
    assert all(abs(100 * r.changed / 5000 - v) < 0.1 for r, v in zip(rows, [9.0, 3.5, 0.0, 3.3, 9.0]))
    assert [de(r.K, 2, True) for r in rows if r.sum_k != 1.0] == ["+3,72", "+0,73", "−0,40", "−0,70"]
    per_seed = {s: SW.form_table(5000, 20, 5, s, (0.5, 1.0, 1.5)) for s in SEEDS}
    lo05, hi05 = min(v[0].changed for v in per_seed.values()), max(v[0].changed for v in per_seed.values())
    lo15, hi15 = min(v[2].changed for v in per_seed.values()), max(v[2].changed for v in per_seed.values())
    assert abs(lo05 - 440) <= BAND and abs(hi05 - 469) <= BAND
    assert abs(lo15 - 441) <= BAND and abs(hi15 - 481) <= BAND
    assert all(v[1].changed == 0 for v in per_seed.values())
    assert abs((5000 - rows[0].changed) - 4552) <= BAND and all(r.ties == 0 for r in rows)
    assert (de(rows[0].tau_mean, 3), de(rows[0].tau_min, 3), de(rows[4].tau_mean, 3), de(rows[4].tau_min, 3)) == ("0,950", "0,842", "0,960", "0,863")
    assert SW.standard_error_points(448, 5000) < 0.7
    _readme_has("**448 / 177 / 0 / 166 / 451**", "(9,0 / 3,5 / 0,0 / 3,3 / 9,0 %)", "+3{,}72$ / +0,73 / 0 / −0,40 / −0,70", "440 bis 469 (8,8 bis 9,4 %)", "441 bis 481 (8,8 bis 9,6 %)", "4552 von 5000",
                "Mittel 0,950 und kleinster Wert 0,842", "0,960 und 0,863", "höchstens 0,7 Prozentpunkte")
    assert de(100 * 440 / 5000, 1) == "8,8" and de(100 * 469 / 5000, 1) == "9,4" and de(100 * 441 / 5000, 1) == "8,8" and de(100 * 481 / 5000, 1) == "9,6"


# --- B6 --------------------------------------------------------------------------------------------------------------------------------

def test_b6_rechenfalle():
    cont = SW.continuity_probe(5000, 20, 5, SEED)
    assert ev.sci(cont["K_stable"]) == "2,610e-06" and ev.sci(cont["K_series"]) == "2,610e-06" and ev.sci(cont["K_naive"]) == "1,000e-12"
    assert cont["changed_stable"] == 0 and abs(cont["changed_naive"] - 2) <= BAND and cont["changed_naive"] > cont["changed_stable"]
    _readme_has("2,610e-06 (Näherung $(1 - \\Sigma k)/e_2$: 2,610e-06)", "1,000e-12", "**2 von 5000**")


# --- B7 --------------------------------------------------------------------------------------------------------------------------------

def test_b7_form_nach_m_und_n():
    table = {}
    for m in (2, 5, 8):
        for n in (3, 10, 40):
            r = SW.form_table(3000, n, m, SEED, (0.5, 1.5))
            table[(m, n)] = (r[0].changed, r[1].changed)
    want = {(2, 3): (209, 254), (2, 10): (304, 680), (2, 40): (292, 882), (5, 3): (121, 106), (5, 10): (220, 217), (5, 40): (300, 335), (8, 3): (111, 77), (8, 10): (189, 178), (8, 40): (257, 233)}
    for key, (a, b) in want.items():
        _band(table[key], (a, b))
    rising = 0
    for m in (2, 5, 8):
        for j in (0, 1):
            seq = [table[(m, n)][j] for n in (3, 10, 40)]
            rising += int(seq[0] < seq[1] < seq[2])
    assert rising == 5 and not (table[(2, 3)][0] < table[(2, 10)][0] < table[(2, 40)][0])
    _readme_has("$n = 3$: 209 / 254, $n = 10$: 304 / 680, $n = 40$: 292 / 882", "121 / 106, 220 / 217, 300 / 335", "111 / 77, 189 / 178, 257 / 233", "in allen sechs Zeilen", "in fünf der sechs Zeilen", "(209, 304, 292)")


# --- B8 --------------------------------------------------------------------------------------------------------------------------------

def test_b8_mittelwerte_einsetzen():
    rows = SW.risk_table(5000, 2, SEED, C.FORM_ADD)
    assert [r.rho_norm for r in rows] == [4.0, 2.0, 1.0, 0.5, 0.25, 0.1]
    _band([r.changed for r in rows], [121, 247, 495, 925, 1628, 2399])
    assert all(
        abs(100 * r.changed / 5000 - v) < 0.1 for r, v in zip(rows, [2.4, 4.9, 9.9, 18.5, 32.6, 48.0]))
    per_seed = {s: SW.risk_table(5000, 2, s, C.FORM_ADD) for s in SEEDS}
    for idx, (lo, hi) in ((1, (238, 272)), (5, (2366, 2442))):
        counts = [v[idx].changed for v in per_seed.values()]
        assert abs(min(counts) - lo) <= BAND and abs(max(counts) - hi) <= BAND
    X = SW.risk_values(5000, 2, SEED)
    assert SW.risk_row(X, math.inf, C.FORM_ADD).changed == 0
    mult = SW.risk_table(5000, 2, SEED, C.FORM_MULT)
    _band([r.changed for r in mult], [398, 448, 574, 954, 1608, 2307])
    assert abs(SW.risk_row(X, math.inf, C.FORM_MULT).changed - 392) <= BAND
    assert abs(mult[3].changed / 5000 * 100 - 19.1) < 0.2
    for rho, want in ((0.5, [552, 600, 608, 598]), (2.0, [143, 159, 142, 176])):
        _band([r.changed for _, r in SW.risk_by_scenarios(3000, SEED, rho, C.FORM_ADD)], want)
    for s in SEEDS:
        by = {S_: r.changed for S_, r in SW.risk_by_scenarios(3000, s, 0.5, C.FORM_ADD)}
        assert by[10] > by[2], s
    assert de(552 / 3000 * 100, 1) == "18,4" and de(608 / 3000 * 100, 1) == "20,3" and de(C.RHO_STAR_SINGLE / C.GAIN_SPAN, 2) == "1,85"
    _readme_has("**121 / 247 / 495 / 925 / 1628 / 2399**", "(2,4 / 4,9 / 9,9 / 18,5 / 32,6 / 48,0 %)", "238 bis 272 (4,8 bis 5,4 %)", "2366 bis 2442 (47,3 bis 48,8 %)", "0 von 5000",
                "398 / 448 / 574 / 954 / 1608 / 2307", "**bereits bei linearem Einzelnutzen 392 von 5000**", "552 / 600 / 608 / 598 (18,4 bis 20,3 %)", "143 / 159 / 142 / 176", "in allen fünf Zufalls-Seeds",
                "555,8/300 = 1,85", "954 von 5000, 19,1 %", "2,4 % bis 48,0 %")
    assert de(100 * 238 / 5000, 1) == "4,8" and de(100 * 272 / 5000, 1) == "5,4" and de(100 * 2366 / 5000, 1) == "47,3" and de(100 * 2442 / 5000, 1) == "48,8"


# --- B9 --------------------------------------------------------------------------------------------------------------------------------

def test_b9_kopplung():
    rows = SW.coupling_table(5000, 10, SEED)
    assert [r.sum_k for r in rows] == [0.5, 0.8, 1.0, 1.2, 1.5]
    _band([r.same_vs_opposite[0] for r in rows], [570, 255, 0, 270, 650])
    _band([r.same_vs_independent[0] for r in rows], [327, 125, 0, 130, 289])
    _band([r.opposite_vs_independent[0] for r in rows], [258, 134, 0, 144, 370])
    assert all(r.add_same_vs_opposite == (0, 0) for r in rows) and rows[2].same_vs_opposite[0] == 0 and rows[2].add_vs_mult_same[0] == 0
    assert all(r.same_vs_opposite[1] == r.same_vs_independent[1] == r.opposite_vs_independent[1] == 0 for r in rows)
    assert all(
        abs(100 * r.same_vs_opposite[0] / 5000 - v) < 0.1 for r, v in zip(rows, [11.4, 5.1, 0.0, 5.4, 13.0]))
    per_seed = {s: SW.coupling_table(5000, 10, s, (0.5, 1.5)) for s in SEEDS}
    for idx, (lo, hi) in ((0, (506, 586)), (1, (595, 701))):
        counts = [v[idx].same_vs_opposite[0] for v in per_seed.values()]
        assert abs(min(counts) - lo) <= BAND and abs(max(counts) - hi) <= BAND
    assert abs(rows[0].diff_mean - 0.0277) < 5e-4 and abs(rows[0].diff_max - 0.2254) < 5e-4 and rows[0].diff_min >= -1e-12
    assert rows[4].diff_max <= 1e-12 and abs(rows[4].diff_min + 0.2254) < 5e-4 and rows[0].total * 10 == 50000
    assert (de(rows[0].diff_mean, 4, True), de(rows[0].diff_max, 4, True)) == ("+0,0277", "+0,2254")
    assert de(100 * 506 / 5000, 1) == "10,1" and de(100 * 586 / 5000, 1) == "11,7" and de(100 * 595 / 5000, 1) == "11,9" and de(100 * 701 / 5000, 1) == "14,0"
    _readme_has("**Additiv ändert sich Platz 1 mit der Kopplung in 0 von 5000 Problemen**", "**570 / 255 / 0 / 270 / 650**", "(11,4 / 5,1 / 0,0 / 5,4 / 13,0 %)", "327 / 125 / 0 / 130 / 289", "258 / 134 / 0 / 144 / 370",
                "506 bis 586 (10,1 bis 11,7 %)", "595 bis 701 (11,9 bis 14,0 %)", "Mittel +0,0277, größter Unterschied +0,2254", "in allen 50 000 Alternativen", "in 0 Fällen verschieden")


def test_b9_unterschied_der_kopplungen_zaehlt_ohne_gleichstaende_in_allen_laeufen():
    """README: „In keinem der Läufe dieser Tabelle trat ein Gleichstand auf“ (Zufalls-Seed 1017, Form, Risiko additiv und multiplikativ, Kopplung)."""
    ties = sum(r.ties for r in SW.form_table(5000, 20, 5, SEED)) + sum(r.ties for r in SW.risk_table(5000, 2, SEED, C.FORM_ADD)) + sum(r.ties for r in SW.risk_table(5000, 2, SEED, C.FORM_MULT))
    ties += sum(sum((r.same_vs_opposite[1], r.same_vs_independent[1], r.opposite_vs_independent[1], r.add_vs_mult_same[1])) for r in SW.coupling_table(5000, 10, SEED))
    assert ties == 0
    _readme_has("In keinem der Läufe dieser Tabelle trat ein Gleichstand der beiden besten Werte auf")


def test_rechenzeit_am_reglermaximum_laeuft_durch():
    """README: etwa 0,3 bis 0,6 Sekunden je Experiment am Reglermaximum (Messung, nicht festgeschrieben): hier ohne Zeitbehauptung, nur dass die größten Fälle laufen."""
    assert len(SW.form_table(5000, 40, 8, 5)) == 5
    assert len(SW.risk_table(5000, 10, 5, C.FORM_MULT)) == 6
    assert len(SW.risk_by_scenarios(5000, 5, 0.5, C.FORM_MULT)) == 4
    assert len(SW.coupling_table(5000, 40, 5)) == 5
    cont = SW.continuity_probe(5000, 40, 8, 5)
    assert cont["changed_stable"] == 0
    _readme_has("etwa 0,3 bis 0,6 Sekunden")


def test_nenner_und_zaehleinheiten_stehen_in_der_readme():
    _readme_has("Zähleinheit: Matrizen", "Probleme (Risiko, Kopplung)", "Zufalls-Seed 1017", "ausgedacht", "Modell, kein Fall", "Skalierungskonstanten", "nicht „Gewichte“")
