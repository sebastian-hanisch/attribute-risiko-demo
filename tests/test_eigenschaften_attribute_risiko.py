"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für die Formen gelten müssen, ohne dass man das richtige Ergebnis kennt.
Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed: Umnummerieren, Spiegeln der Ziele, Ecken und Summe 1, Linearität in den Wahrscheinlichkeiten,
Vorzeichen von c und Stetigkeit bei Σk → 1 (Regressionstest der Rechenfalle)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U


def test_eigenschaft_umnummerieren_aendert_nur_die_namen():
    u = SW.form_values(40, 9, 4, 3)
    perm = np.random.default_rng(0).permutation(9)
    k = U.harmonic_weights(4) * 0.7
    assert np.allclose(U.mult_utility(u[:, perm, :], k), U.mult_utility(u, k)[:, perm], atol=1e-14)
    assert np.allclose(U.add_utility(u[:, perm, :], k), U.add_utility(u, k)[:, perm], atol=1e-14)


def test_eigenschaft_spiegeln_der_attribute_aendert_nichts():
    u = SW.form_values(40, 9, 2, 4)
    k = np.array([0.45, 0.2])
    assert np.allclose(U.mult_utility(u, k), U.mult_utility(u[..., ::-1], k[::-1]), atol=1e-14)


def test_eigenschaft_ecken_und_summe_eins():
    rng = np.random.default_rng(1)
    for _ in range(40):
        m = int(rng.integers(2, 7))
        k = rng.uniform(0.05, 0.4, m)
        assert U.mult_utility(np.ones(m), k) == pytest.approx(1.0, abs=1e-12)           # U(1, …, 1) = 1 für jede Summe
        assert U.mult_utility(np.zeros(m), k) == pytest.approx(0.0, abs=1e-12)
        k1 = k / k.sum()                                                               # Σk = 1: multiplikativ = additiv
        u = rng.uniform(size=m)
        assert U.mult_utility(u, k1) == pytest.approx(float(u @ k1), abs=1e-12)
        i = int(rng.integers(m))
        e = np.zeros(m)
        e[i] = 1.0
        assert U.mult_utility(e, k) == pytest.approx(k[i], abs=1e-12)                    # die k_i sind Nutzen der Ecken


def test_eigenschaft_erwartungsnutzen_linear_in_den_wahrscheinlichkeiten():
    h = (Fr(2, 5), Fr(3, 10), Fr(1, 2), Fr(3, 5))
    for lam in (Fr(1, 4), Fr(1, 2), Fr(3, 4)):
        a = SC.expected_utility(SC.cells("G", Fr(1, 10), math.inf, *h[2:]), *h[:2], "mult")
        b = SC.expected_utility(SC.cells("G", Fr(2, 5), math.inf, *h[2:]), *h[:2], "mult")
        mix = SC.expected_utility(SC.cells("G", lam * Fr(1, 10) + (1 - lam) * Fr(2, 5), math.inf, *h[2:]), *h[:2], "mult")
        assert mix == lam * a + (1 - lam) * b                                            # die Zellen sind in q linear, EU also auch


def test_eigenschaft_vorzeichen_von_c_ordnet_die_kopplungen():
    u = SW.coupling_values(500, 10, 8)
    for sum_k, sign in ((0.5, 1), (0.8, 1), (1.2, -1), (1.5, -1)):
        eu = SW.coupling_utilities(u, sum_k)
        d = eu["gleichläufig"]["mult"] - eu["gegenläufig"]["mult"]
        assert np.all(sign * d >= -1e-12)                                                # c > 0: gleichläufig ≥ gegenläufig; c < 0: umgekehrt


def test_stetigkeit_bei_summe_k_gegen_eins():
    """Regressionstest der Rechenfalle: K für Σk = 0,999999 aus den Koeffizienten der Produktentwicklung, nicht aus der direkten Gleichung."""
    w = U.harmonic_weights(5)
    for s in (0.999999, 1.000001):
        k = w * s
        e2 = sum(k[i] * k[j] for i in range(5) for j in range(i + 1, 5))
        assert U.mult_constant(k) == pytest.approx((1.0 - s) / e2, rel=0.01)
    assert abs(U.mult_constant_naive(w * 0.999999) / U.mult_constant(w * 0.999999) - 1.0) > 0.5   # die direkte Gleichung gibt ein falsches K
    u = SW.form_values(300, 8, 5, 5)
    ua = U.add_utility(u, w)
    assert np.max(np.abs(U.mult_utility(u, w * 0.999999) - ua)) < 1e-4                    # stetig bis Σk = 1
    assert SW.count_top_changes(ua, U.mult_utility(u, w * 0.999999))[0] <= 1


