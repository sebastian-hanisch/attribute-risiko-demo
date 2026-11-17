"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): n = 1, n = 2, Gleichstand, inklusive Grenzen, Extremwerte der Regler."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest

import atr_constants as C
import atr_evaluation as ev
import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U


def test_exactly_two_alternatives():
    """n = 2: der kleinste Fall mit einem Vergleich (Zweige `n < 2` und Division durch n − 1 dürfen nicht greifen); Kendall-Tau hat dann genau ein Paar."""
    a, b = np.array([[0.2, 0.9]]), np.array([[0.7, 0.1]])
    assert SW.count_top_changes(a, b) == (1, 0)
    assert U.kendall_tau(a, b)[0] == -1.0
    row = SW.form_row(SW.form_values(50, 2, 3, 1), 0.5)
    assert row.total == 50 and 0 <= row.changed <= 50 and -1.0 <= row.tau_min <= row.tau_mean <= 1.0


def test_single_alternative_and_single_attribute():
    """n = 1: kein Wechsel möglich, Kendall-Tau undefiniert (NaN); m = 1: die Form ist der Einzelnutzen."""
    assert SW.count_top_changes(np.array([[1.0]]), np.array([[3.0]])) == (0, 0)
    assert np.isnan(U.kendall_tau(np.array([[1.0]]), np.array([[2.0]]))[0])
    u = np.array([0.25, 0.75])
    assert float(U.mult_utility(u[:1], [1.0 - 1e-15])) == pytest.approx(0.25, abs=1e-12)


def test_ties_are_counted_as_ties_not_as_changes():
    """Zwei gleich gute Spitzen (Abstand unter 1e-12) zählen als Gleichstand; sie dürfen nie als Wechsel erscheinen."""
    a = np.array([[0.5, 0.5 + 1e-14, 0.1]])
    b = np.array([[0.6, 0.2, 0.6 + 1e-14]])
    assert SW.count_top_changes(a, b) == (0, 1)


def test_bounds_are_inclusive():
    """Grenzen „von … bis …“ gelten inklusive: q = 0 und q = ½ sind erlaubt, kleiner und größer nicht; Σk = 0,5 und 1,5 und die Reglergrenzen laufen durch."""
    for q in (Fr(0), Fr(1, 2)):
        assert sum(SC.cell_probabilities(q).values()) == 1
    with pytest.raises(ValueError):
        SC.cell_probabilities(Fr(1, 2) + Fr(1, 1000))
    for sk in (C.SUMK_MIN, C.SUMK_MAX):
        assert SW.form_row(SW.form_values(C.P_MIN, C.N_MIN, C.M_MIN, 1), sk).total == C.P_MIN
        assert SW.coupling_result(SW.coupling_values(C.P_MIN, C.N_MIN, 1), sk).total == C.P_MIN
    rows = SW.risk_table(C.P_MIN, C.S_MAX, 1, C.FORM_MULT)
    assert [r.rho_norm for r in rows] == list(C.RHO_NORM_OPTIONS) and all(0 <= r.changed <= r.total for r in rows)


def test_extreme_slider_values_in_the_capacity_case():
    """k an den Reglergrenzen, ρ = 100 und ∞, Servicegrad 0 und 100 %: endliche Werte, kein Fehler, additive und multiplikative Form gleich bei Σk = 1."""
    for k1, k2 in ((C.K_MIN, C.K_MIN), (C.K_MAX, C.K_MAX), (C.K_MIN, C.K_MAX)):
        for rho in (math.inf, 100.0):
            for sg, sk in ((0, 0), (100, 100), (0, 100)):
                hs = ev.HandSettings(U.as_fraction(k1), U.as_fraction(k2), Fr(1, 4), rho, Fr(sg, 100), Fr(sk, 100))
                r = ev.analyse_hand(hs)
                assert all(math.isfinite(float(v)) for v in r.eu.values())
                kind, text = ev.verdict_hand(r)
                assert kind in ("success", "info", "warning") and text
    r = ev.analyse_hand(ev.HandSettings(Fr(2, 5), Fr(3, 5), Fr(1, 4), math.inf, Fr(1, 2), Fr(3, 5)))
    assert r.c == 0 and r.d_mult == r.d_add


def test_servicegrad_hundred_percent_makes_the_second_attribute_irrelevant_for_g():
    """Servicegrad im Engpass 100 % für beide: u2 ≡ 1, es gibt keine Kopplung mehr (EU(G) hängt nicht von q ab, auch multiplikativ)."""
    h = (Fr(2, 5), Fr(3, 10), Fr(1), Fr(1))
    values = {SC.eu("G", q, math.inf, *h, "mult") for q in (Fr(0), Fr(1, 4), Fr(1, 2))}
    assert len(values) == 1


def test_all_alternatives_equal_gives_no_change_and_no_division_by_zero():
    u = np.full((5, 4, 3), 0.5)
    row = SW.form_row(u, 0.5)
    assert row.changed == 0 and row.ties == 5
    cr = SW.coupling_result(np.full((5, 4, 2, 2), 0.5), 1.5)
    assert cr.same_vs_opposite == (0, 5)
