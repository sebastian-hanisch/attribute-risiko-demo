"""Orakel-Tests: die Kernroutinen gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2 und §5, DEMO-PLAYBOOK §12).

Orakel (alle unabhängig vom Rechenweg der Demo):
* Produktform gegen ausmultiplizierte Form: `fractions`, 300 zufällige Bruchpaare (die Demo rechnet mit expm1/log1p über die Koeffizienten-Wurzel von K).
* K für zwei Attribute gegen die geschlossene Form (1 − k1 − k2)/(k1·k2); K für drei Attribute gegen die Wurzel des Polynoms (`numpy.roots`).
* Erwartungsnutzen über die Zellen (aufgezählt) gegen die Kovarianzform (Brüche) und gegen die geschlossene Form im Float.
* ρ* aus u1(30; ρ) = 1/2 gegen ρ* aus dem Sicherheitsäquivalent CE_G(ρ) = 30 (CARA, anderer Rechenweg, Orakel der Art von `kuchen-vergroessern-demo`).
* Linearer Einzelnutzen: Erwartungsnutzen = Nutzen der Erwartungswerte (additiv).
* Die Zufallsproblem-Auswertungen (Kopplung, Form) gegen eine Schleife über Alternativen mit ganzzahligen Werten in `fractions`.
* Kendall-Tau gegen `scipy.stats.kendalltau`; Zellenaufzählung der Kopplung gegen die geschlossene Form c·½·Δa·Δb.
Plattform: Zählungen und Brüche exakt, Gleitkomma mit Toleranz (1e-12 bzw. 1e-6 für die Wurzelsuche)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import stats

import atr_constants as C
import atr_rng as RNG
import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U


def _uni(seed, n):
    return RNG.uniform_array(seed, n)


def _bruch(x, nenner=20):
    return Fr(int(x * (nenner + 1)), nenner)


def test_produktform_gleich_ausmultiplizierter_form_in_bruechen():
    u = _uni(11, 300 * 4)
    done = 0
    for i in range(300):
        a, b = Fr(int(u[4 * i] * 59) + 1, 100), Fr(int(u[4 * i + 1] * 59) + 1, 100)
        if a + b == 1:
            continue
        K = (1 - a - b) / (a * b)
        assert 1 + K == (1 + K * a) * (1 + K * b)
        u1, u2 = _bruch(u[4 * i + 2]), _bruch(u[4 * i + 3])
        assert ((1 + K * a * u1) * (1 + K * b * u2) - 1) / K == U.u2_mult(u1, u2, a, b)
        done += 1
    assert done > 280


def test_k_zwei_attribute_geschlossen_und_drei_attribute_wurzel_des_polynoms():
    rng_vals = _uni(12, 200)
    for i in range(100):
        a, b = 0.05 + 0.6 * rng_vals[2 * i], 0.05 + 0.6 * rng_vals[2 * i + 1]
        if abs(a + b - 1.0) < 1e-6:
            continue
        assert U.mult_constant([a, b]) == pytest.approx((1 - a - b) / (a * b), rel=1e-9)
    k = np.array([0.3, 0.25, 0.2])
    e2, e3 = k[0] * k[1] + k[0] * k[2] + k[1] * k[2], k[0] * k[1] * k[2]
    roots = [r.real for r in np.roots([e3, e2, k.sum() - 1.0]) if abs(r.imag) < 1e-12 and r.real > -1]
    assert len(roots) == 1 and U.mult_constant(k) == pytest.approx(roots[0], abs=1e-9)
    kk = U.mult_constant(k)
    assert 1 + kk == pytest.approx(np.prod(1 + kk * k), rel=1e-12)                      # die definierende Gleichung selbst


def test_erwartungsnutzen_zellen_gegen_kovarianzform_und_geschlossene_form():
    h = {"k1": Fr(2, 5), "k2": Fr(3, 10), "s_g": Fr(1, 2), "s_k": Fr(3, 5)}
    c = 1 - h["k1"] - h["k2"]
    for q in (Fr(0), Fr(1, 10), Fr(1, 4), Fr(2, 5), Fr(1, 2)):
        for alt in SC.ALTS:
            cs = SC.cells(alt, q, math.inf, h["s_g"], h["s_k"])
            e1, e2, cov = SC.moments(cs)
            assert SC.expected_utility(cs, h["k1"], h["k2"], "mult") == h["k1"] * e1 + h["k2"] * e2 + c * (e1 * e2 + cov)
            assert SC.expected_utility(cs, h["k1"], h["k2"], "add") == (h["k1"] * e1 + h["k2"] * e2) / (h["k1"] + h["k2"])
    # Float mit exponentiellem Gewinnnutzen: Zellenaufzählung gegen die zu Fuß geschriebene Formel
    for rho in (100.0, 555.8, 2000.0):
        for q in (0.0, 0.25, 0.5):
            u1h, u1l, u130 = (float(U.gain_utility(g, rho, -100, 200)) for g in (200, -100, 30))
            k1, k2 = 0.4, 0.3
            f = lambda a, b: k1 * a + k2 * b + (1 - k1 - k2) * a * b  # noqa: E731
            g_hand = (0.5 - q) * f(u1h, 1.0) + q * f(u1h, 0.5) + q * f(u1l, 1.0) + (0.5 - q) * f(u1l, 0.5)
            k_hand = 0.5 * f(u130, 1.0) + 0.5 * f(u130, 0.6)
            assert float(SC.eu("G", q, rho, k1, k2, 0.5, 0.6, "mult")) == pytest.approx(g_hand, abs=1e-12)
            assert float(SC.eu("K", q, rho, k1, k2, 0.5, 0.6, "mult")) == pytest.approx(k_hand, abs=1e-12)


def test_kipp_risikotoleranz_zwei_rechenwege():
    rho_u = SC.rho_star_single()                                                         # u1(30; ρ) = 1/2
    from scipy import optimize
    rho_ce = optimize.brentq(lambda r: SC.ce_gain_g(r) - 30.0, 50.0, 1e4, xtol=1e-12)    # Sicherheitsäquivalent = 30
    assert rho_u == pytest.approx(rho_ce, abs=1e-6) and rho_u == pytest.approx(C.RHO_STAR_SINGLE, abs=1e-5)
    assert rho_u == pytest.approx(555.801, abs=1e-3)


def test_linearer_einzelnutzen_erwartungsnutzen_gleich_nutzen_der_erwartungswerte():
    X = SW.draw(5, 9, (200, 6, 4, 3))                                                    # (Probleme, Alternativen, Szenarien, Attribute)
    w = U.harmonic_weights(3)
    assert np.max(np.abs((U.single_utility_exp(X, math.inf) @ w).mean(axis=2) - X.mean(axis=2) @ w)) < 1e-12
    for rho in (4.0, 0.5):                                                               # gekrümmt: Jensen, der Erwartungsnutzen liegt unter dem Nutzen der Mittel
        eu = (U.single_utility_exp(X, rho) @ w).mean(axis=2)
        assert np.all(eu <= U.single_utility_exp(X.mean(axis=2), rho) @ w + 1e-12)


def test_zufallsprobleme_gegen_schleife_ueber_alternativen_in_bruechen():
    """200 Probleme, ganzzahlige Werte (Fractions), zwei Attribute: Platz 1 bei der Form (sichere Ausgänge) und bei den Kopplungen gegen die Schleife."""
    P, n = 200, 6
    vals = RNG.uniform_array(77, P * n * 4).reshape(P, n, 2, 2)
    q = np.round(vals * 20) / 20.0                                                       # Werte aus {0, 1/20, …, 1}: exakt als Bruch
    for sum_k in (0.5, 1.5):
        k1, k2 = Fr(3, 5) * Fr(sum_k).limit_denominator(10), Fr(2, 5) * Fr(sum_k).limit_denominator(10)
        c = 1 - k1 - k2
        res = SW.coupling_utilities(q, sum_k)
        flips = {("gleichläufig", "gegenläufig"): 0}
        for p in range(P):
            eu = {"gleichläufig": [], "gegenläufig": []}
            for i in range(n):
                a = [Fr(int(round(v * 20)), 20) for v in q[p, i, :, 0]]
                b = [Fr(int(round(v * 20)), 20) for v in q[p, i, :, 1]]
                ah, at_, bh, bt = max(a), min(a), max(b), min(b)
                eu["gleichläufig"].append(sum(Fr(1, 2) * U.u2_mult(x, y, k1, k2) for x, y in ((ah, bh), (at_, bt))))
                eu["gegenläufig"].append(sum(Fr(1, 2) * U.u2_mult(x, y, k1, k2) for x, y in ((ah, bt), (at_, bh))))
                assert float(eu["gleichläufig"][i]) == pytest.approx(res["gleichläufig"]["mult"][p, i], abs=1e-12)
                assert float(eu["gegenläufig"][i]) == pytest.approx(res["gegenläufig"]["mult"][p, i], abs=1e-12)
                # Unterschied = c · 1/2 · Spanne(a) · Spanne(b), exakt
                assert eu["gleichläufig"][i] - eu["gegenläufig"][i] == c * Fr(1, 2) * (ah - at_) * (bh - bt)
            top = {k: max(range(n), key=lambda i: eu[k][i]) for k in eu}
            gap = sorted(eu["gleichläufig"])[-1] - sorted(eu["gleichläufig"])[-2]
            gap2 = sorted(eu["gegenläufig"])[-1] - sorted(eu["gegenläufig"])[-2]
            if gap > Fr(1, 10**9) and gap2 > Fr(1, 10**9) and top["gleichläufig"] != top["gegenläufig"]:
                flips[("gleichläufig", "gegenläufig")] += 1
        assert flips[("gleichläufig", "gegenläufig")] == SW.count_top_changes(res["gleichläufig"]["mult"], res["gegenläufig"]["mult"])[0]


def test_form_platz_eins_gegen_schleife_mit_bruechen():
    """Zwei Attribute, k rational: Platz 1 additiv gegen multiplikativ, Schleife in Brüchen gegen die vektorisierte Zählung (200 Matrizen)."""
    P, n = 200, 7
    u = np.round(RNG.uniform_array(78, P * n * 2).reshape(P, n, 2) * 20) / 20.0
    w = U.harmonic_weights(2)                                                           # (0,6; 0,4)
    for sum_k in (0.5, 1.5):
        ks = Fr(sum_k).limit_denominator(10)
        k1, k2 = Fr(3, 5) * ks, Fr(2, 5) * ks
        want = 0
        for p in range(P):
            us = [(Fr(int(round(u[p, i, 0] * 20)), 20), Fr(int(round(u[p, i, 1] * 20)), 20)) for i in range(n)]
            add = [U.u2_add(a, b, k1, k2) for a, b in us]
            mul = [U.u2_mult(a, b, k1, k2) for a, b in us]
            sa, sm = sorted(add), sorted(mul)
            if sa[-1] - sa[-2] > Fr(1, 10**9) and sm[-1] - sm[-2] > Fr(1, 10**9) and max(range(n), key=lambda i: add[i]) != max(range(n), key=lambda i: mul[i]):
                want += 1
        ua, um = U.add_utility(u, w * sum_k), U.mult_utility(u, w * sum_k)
        assert SW.count_top_changes(ua, um)[0] == want


def test_kendall_tau_gegen_scipy():
    a, b = _uni(21, 40 * 12).reshape(40, 12), _uni(22, 40 * 12).reshape(40, 12)
    ours = U.kendall_tau(a, b)
    ref = np.array([stats.kendalltau(a[i], b[i])[0] for i in range(40)])
    assert np.max(np.abs(ours - ref)) < 1e-12


def test_additiv_haengt_nicht_von_der_kopplung_ab_exakt():
    """300 Probleme in Brüchen: bei gleichen Rändern ist der additive Erwartungsnutzen bei gleichläufiger und gegenläufiger Kopplung gleich."""
    k1, k2 = Fr(2, 5), Fr(3, 10)
    vals = _uni(33, 300 * 4)
    for p in range(300):
        a = [Fr(int(vals[4 * p + j] * 21), 20) for j in range(2)]
        b = [Fr(int(vals[4 * p + 2 + j] * 21), 20) for j in range(2)]
        ah, at_, bh, bt = max(a), min(a), max(b), min(b)
        add_g = sum(Fr(1, 2) * (k1 * x + k2 * y) for x, y in ((ah, bh), (at_, bt)))
        add_x = sum(Fr(1, 2) * (k1 * x + k2 * y) for x, y in ((ah, bt), (at_, bh)))
        assert add_g == add_x


# --- Randfälle -------------------------------------------------------------------------------------------------------------------------------

def test_randfaelle_nutzen():
    with pytest.raises(ValueError):
        U.gain_utility(10.0, 100.0, 5.0, 5.0)                                            # Spanne 0: nicht definierbar, nicht durch null teilen
    k = np.array([0.3, 0.2, 0.0])                                                        # k_i = 0: Attribut ohne Einfluss
    u = SW.form_values(20, 5, 3, 6)
    assert np.allclose(U.mult_utility(u, k), U.mult_utility(u[..., :2], k[:2]), atol=1e-14)
    kk = np.array([0.99, 0.99])                                                          # Σk nahe 2: K nahe −1, Wurzel im Intervall (−1, 0)
    K = U.mult_constant(kk)
    assert -1.0 < K < 0.0 and 1 + K == pytest.approx(np.prod(1 + K * kk), rel=1e-9)
    assert U.mult_constant([1.0 - 1e-15]) == 0.0                                          # m = 1 mit k ≈ 1: K = 0
    assert float(U.mult_utility(np.array([0.3]), [1.0 - 1e-15])) == pytest.approx(0.3, abs=1e-12)
    for bad in ([0.5], [1.0, 0.2], [-0.1, 0.5]):
        with pytest.raises(ValueError):
            U.mult_constant(bad)
    # ρ → ∞: gekrümmter Einzelnutzen nähert sich dem linearen, ρ → 0: bleibt endlich und in [0, 1]
    assert float(U.single_utility_exp(0.3, 1e12)) == pytest.approx(0.3, abs=1e-9)
    tiny = U.single_utility_exp(np.array([0.0, 0.5, 1.0]), 1e-3)
    assert np.all(np.isfinite(tiny)) and tiny[0] == 0.0 and tiny[-1] == pytest.approx(1.0)


def test_randfaelle_kopplung_und_zaehlung():
    p = SC.cell_probabilities(Fr(0))
    assert p[("hoch", "Engpass")] == 0 and p[("niedrig", "normal")] == 0 and sum(p.values()) == 1          # Fréchet-Grenze q = 0
    p = SC.cell_probabilities(Fr(1, 2))
    assert p[("hoch", "normal")] == 0 and p[("niedrig", "Engpass")] == 0 and sum(p.values()) == 1          # q = 1/2
    for bad in (Fr(-1, 100), Fr(51, 100)):
        with pytest.raises(ValueError):
            SC.cell_probabilities(bad)
    same = np.ones((3, 4)) * 0.5                                                                           # alle Alternativen gleich: Gleichstand, kein Wechsel
    assert SW.count_top_changes(same, same + 1e-14) == (0, 3)
    assert SW.count_top_changes(np.array([[1.0]]), np.array([[2.0]])) == (0, 0)                             # n = 1: nur eine Alternative
    assert SW.count_top_changes(np.array([[1.0, 2.0]]), np.array([[2.0, 1.0]])) == (1, 0)                   # n = 2: der kleinste Fall mit Wechsel
