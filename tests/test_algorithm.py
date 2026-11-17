"""Einheiten `atr_utility` und `atr_scenario` gegen Handrechnung in Brüchen (Mini-Instanzen, DEMO-PLAYBOOK §1b).

Handbeispiel: k = (2/5, 3/10): 1 + K = (1 + K·2/5)(1 + K·3/10) gibt K = (1 − 7/10)/(3/25) = 5/2; die Faktoren sind 2 und 7/4, 2·7/4 = 7/2.
U = 2/5·u1 + 3/10·u2 + 3/10·u1·u2. Ecken U(1,1) = 1, U(0,0) = 0, U(1,0) = 2/5, U(0,1) = 3/10; Mittelpunkt (1/2, 1/2): multiplikativ 17/40, additiv mit den
unnormierten k 7/20."""

import math
from fractions import Fraction as Fr
from itertools import product

import numpy as np
import pytest

import atr_scenario as SC
import atr_utility as U

K1, K2 = Fr(2, 5), Fr(3, 10)


def test_skalierungskonstante_von_hand():
    K = (1 - K1 - K2) / (K1 * K2)
    assert K == Fr(5, 2)
    assert (1 + K * K1, 1 + K * K2) == (2, Fr(7, 4)) and 1 + K == (1 + K * K1) * (1 + K * K2)
    assert U.mult_constant([0.4, 0.3]) == pytest.approx(2.5, rel=1e-12)
    assert U.cross_coefficient(K1, K2) == Fr(3, 10) == Fr(5, 2) * K1 * K2


def test_ecken_und_mittelpunkt():
    one, zero, half = Fr(1), Fr(0), Fr(1, 2)
    assert (U.u2_mult(one, one, K1, K2), U.u2_mult(zero, zero, K1, K2)) == (1, 0)
    assert (U.u2_mult(one, zero, K1, K2), U.u2_mult(zero, one, K1, K2)) == (K1, K2)     # die k_i sind Nutzen der Ecken
    assert U.u2_mult(half, half, K1, K2) == Fr(17, 40)
    assert K1 * half + K2 * half == Fr(7, 20)
    assert U.u2_add(half, half, K1, K2) == half                                         # normiert: 1/2


def test_summe_eins_macht_multiplikativ_gleich_additiv():
    for a in (Fr(1, 5), Fr(2, 5), Fr(1, 2), Fr(9, 10)):
        for u1, u2 in product((Fr(0), Fr(1, 3), Fr(1, 2), Fr(1)), repeat=2):
            assert U.u2_mult(u1, u2, a, 1 - a) == a * u1 + (1 - a) * u2
    assert U.mult_constant([0.4, 0.6]) == 0.0


def test_gleiche_k_anderer_sieger():
    """Σk = 7/10: A = (1, 0), B = (1/2, 1/2): additiv A (4/7 gegen 1/2), multiplikativ B (2/5 gegen 17/40). Σk = 13/10: umgekehrt."""
    a, b = (Fr(1), Fr(0)), (Fr(1, 2), Fr(1, 2))
    assert (U.u2_add(*a, K1, K2), U.u2_add(*b, K1, K2)) == (Fr(4, 7), Fr(1, 2))
    assert (U.u2_mult(*a, K1, K2), U.u2_mult(*b, K1, K2)) == (Fr(2, 5), Fr(17, 40))
    ka, kb = Fr(7, 10), Fr(3, 5)
    b2 = (Fr(3, 5), Fr(3, 5))
    assert (U.u2_mult(*a, ka, kb), U.u2_mult(*b2, ka, kb)) == (Fr(7, 10), Fr(84, 125))
    assert (U.u2_add(*a, ka, kb), U.u2_add(*b2, ka, kb)) == (Fr(7, 13), Fr(3, 5))
    assert (1 - ka - kb) / (ka * kb) == Fr(-5, 7)


def test_kapazitaetsfall_handwerte(hand):
    """u1(30) = 13/30. G bei q = 0: Zelle (hoch, normal) mit 1/2: U = 1; (niedrig, Engpass) mit 1/2: u = (0, 1/2), U = 3/20: EU(G) = 23/40.
    K: normal U = 181/300, Engpass U = 647/1500: EU(K) = 194/375. Additiv (Gewichte 4/7, 3/7): EU(G) = 17/28, EU(K) = 62/105, Unterschied 1/60."""
    h = hand
    assert SC.u1_of(30, math.inf) == Fr(13, 30)
    eu = lambda alt, q, form: SC.eu(alt, q, h["rho"], h["k1"], h["k2"], h["s_g"], h["s_k"], form)  # noqa: E731
    assert eu("G", Fr(0), "mult") == Fr(23, 40) and eu("K", Fr(0), "mult") == Fr(194, 375)
    assert eu("G", Fr(0), "add") == Fr(17, 28) and eu("K", Fr(0), "add") == Fr(62, 105)
    assert eu("G", Fr(0), "add") - eu("K", Fr(0), "add") == Fr(1, 60)
    for q in (Fr(0), Fr(1, 8), Fr(1, 4), Fr(1, 2)):
        assert eu("K", q, "mult") == Fr(194, 375)                       # K hat einen konstanten Gewinn: keine Kopplung
        assert eu("G", q, "add") == Fr(17, 28)                          # additiv: nur die Ränder zählen
        assert eu("G", q, "mult") == Fr(23, 40) - Fr(3, 20) * q         # Steigung −c·(1 − 1/2) = −3/20
    assert eu("G", Fr(1, 4), "mult") == Fr(43, 80) and eu("G", Fr(1, 2), "mult") == Fr(1, 2)


def test_kipp_kopplung_von_hand(hand):
    h = hand
    qs = SC.tipping_q(h["rho"], h["k1"], h["k2"], h["s_g"], h["s_k"])
    assert qs == (Fr(23, 40) - Fr(194, 375)) * Fr(20, 3) == Fr(173, 450)
    d = lambda q: SC.difference(q, h["rho"], h["k1"], h["k2"], h["s_g"], h["s_k"], "mult")  # noqa: E731
    assert d(qs) == 0 and d(Fr(3, 8)) > 0 > d(Fr(1, 2))


def test_kovarianzform_und_unabhaengigkeit(hand):
    h = hand
    c = 1 - h["k1"] - h["k2"]
    for q in (Fr(0), Fr(1, 8), Fr(1, 4), Fr(3, 8), Fr(1, 2)):
        cs = SC.cells("G", q, h["rho"], h["s_g"], h["s_k"])
        e1, e2, cov = SC.moments(cs)
        assert (e1, e2) == (Fr(1, 2), Fr(3, 4)) and cov == Fr(1, 8) - q / 2
        assert SC.expected_utility(cs, h["k1"], h["k2"], "mult") == h["k1"] * e1 + h["k2"] * e2 + c * (e1 * e2 + cov)
    assert Fr(1, 8) - Fr(1, 4) / 2 == 0                                  # q = 1/4: unabhängig


def test_gleiche_raender_andere_kopplung_zwei_zweipunktlose():
    gleich = [(Fr(1), Fr(1)), (Fr(0), Fr(0))]
    gegen = [(Fr(1), Fr(0)), (Fr(0), Fr(1))]
    eu_g = sum(Fr(1, 2) * U.u2_mult(x, y, K1, K2) for x, y in gleich)
    eu_x = sum(Fr(1, 2) * U.u2_mult(x, y, K1, K2) for x, y in gegen)
    assert (eu_g, eu_x) == (Fr(1, 2), Fr(7, 20)) and eu_g - eu_x == Fr(3, 20)           # c/2
    assert sum(Fr(1, 2) * U.u2_add(x, y, K1, K2) for x, y in gleich) == sum(Fr(1, 2) * U.u2_add(x, y, K1, K2) for x, y in gegen) == Fr(1, 2)
    ka, kb = Fr(7, 10), Fr(3, 5)                                                         # c < 0: gegenläufig besser
    assert sum(Fr(1, 2) * U.u2_mult(x, y, ka, kb) for x, y in gegen) > sum(Fr(1, 2) * U.u2_mult(x, y, ka, kb) for x, y in gleich)


def test_risiko_im_attribut_handwert():
    """Gewinn allein, ρ = 100 GE: Sicherheitsäquivalent von G = −100·ln((e⁻² + e¹)/2) = −35,544 GE (unter 30, der Mittelwert 50 liegt darüber); ρ = 1000: 38,79 über 30."""
    assert SC.ce_gain_g(100.0) == pytest.approx(-100.0 * math.log((math.exp(-2.0) + math.exp(1.0)) / 2.0), abs=1e-9)
    assert SC.ce_gain_g(100.0) == pytest.approx(-35.544017, abs=1e-5)
    assert SC.ce_gain_g(100.0) < 30.0 < 50.0
    assert SC.ce_gain_g(1000.0) == pytest.approx(38.79, abs=0.01) and SC.ce_gain_g(1000.0) > 30.0


def test_normierung_einzelnutzen():
    assert float(U.gain_utility(-100, 100.0, -100, 200)) == 0.0 and float(U.gain_utility(200, 100.0, -100, 200)) == pytest.approx(1.0, abs=1e-15)
    assert float(U.gain_utility(30, math.inf, -100, 200)) == pytest.approx(13 / 30, abs=1e-15)
    assert U.harmonic_weights(2).tolist() == pytest.approx([0.6, 0.4]) and U.harmonic_weights(5).sum() == pytest.approx(1.0)


def test_kendall_tau_von_hand():
    a = np.array([[1.0, 2.0, 3.0]])
    assert U.kendall_tau(a, a)[0] == 1.0 and U.kendall_tau(a, -a)[0] == -1.0
    assert U.kendall_tau(a, np.array([[1.0, 3.0, 2.0]]))[0] == pytest.approx(1 / 3)     # 3 Paare: 2 gleich, 1 anders: (2 − 1)/3
