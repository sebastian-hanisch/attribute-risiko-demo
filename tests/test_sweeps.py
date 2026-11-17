"""Zufallsprobleme: gleicher Zufalls-Seed gleiche Ausgabe, eigene Ströme je Experiment, Σk = 1 gibt 0, Anteile und Zählungen in gültigen Bereichen, Reihenfolge der Σk-Tabelle."""

import numpy as np

import atr_constants as C
import atr_sweeps as SW


def test_same_seed_same_output_and_other_seed_other_output():
    a = SW.form_table(1000, 8, 3, 5)
    assert a == SW.form_table(1000, 8, 3, 5)
    assert a != SW.form_table(1000, 8, 3, 6)
    assert SW.coupling_table(1000, 8, 5) == SW.coupling_table(1000, 8, 5)
    assert SW.risk_table(1000, 2, 5, C.FORM_ADD) == SW.risk_table(1000, 2, 5, C.FORM_ADD)


def test_streams_are_separate_per_experiment():
    f = SW.draw(5, SW.SALT_FORM, (4, 4))
    k = SW.draw(5, SW.SALT_KOPPLUNG, (4, 4))
    assert not np.array_equal(f, k)


def test_more_matrices_extend_the_same_stream():
    """Gleiche Form (n, m), mehr Matrizen: die ersten Matrizen bleiben dieselben (Strom von vorn)."""
    small, big = SW.form_values(1000, 6, 4, 9), SW.form_values(3000, 6, 4, 9)
    assert np.array_equal(small, big[:1000])


def test_sum_k_one_gives_zero_changes_in_every_experiment():
    row = SW.form_table(5000, 20, 5, C.DEFAULT_SEED, (1.0,))[0]
    assert (row.changed, row.ties, row.total) == (0, 0, 5000) and row.tau_mean == 1.0 and row.tau_min == 1.0
    kopp = SW.coupling_table(5000, 10, C.DEFAULT_SEED, (1.0,))[0]
    assert kopp.same_vs_opposite == kopp.same_vs_independent == kopp.opposite_vs_independent == kopp.add_vs_mult_same == (0, 0)


def test_additiv_haengt_nicht_von_der_kopplung_ab_in_jeder_zeile():
    for r in SW.coupling_table(2000, 10, 3):
        assert r.add_same_vs_opposite == (0, 0)


def test_linear_additive_risk_gives_zero_changes_and_multiplicative_does_not():
    """ρ = ∞ (linearer Einzelnutzen): additiv ist der Erwartungsnutzen der Nutzen der Erwartungswerte, 0 Wechsel. Multiplikativ nicht: die Kopplung der
    Szenarien (Erwartung eines Produkts gegen Produkt der Erwartungen) wirkt auch ohne Krümmung."""
    X = SW.risk_values(2000, 2, 7)
    assert SW.risk_row(X, float("inf"), C.FORM_ADD).changed == 0
    assert SW.risk_row(X, float("inf"), C.FORM_MULT).changed > 0


def test_risk_share_grows_with_curvature_additive():
    rows = SW.risk_table(3000, 2, C.DEFAULT_SEED, C.FORM_ADD)
    shares = [r.changed for r in rows]                                              # ρ/Spanne 4, 2, 1, 0,5, 0,25, 0,1
    assert shares == sorted(shares) and shares[-1] > shares[0] * 5


def test_example_matrix_and_continuity_probe_shapes():
    idx, found, add_top, mult_top = SW.example_matrix(2000, 10, 4, 3, 0.5)
    assert 0 <= idx < 2000 and found and len(add_top) == len(mult_top) == 5
    assert add_top[0][0] != mult_top[0][0]                                        # die gefundene Matrix hat verschiedenen Platz 1
    assert add_top[0][1] >= add_top[1][1] >= add_top[2][1]
    cont = SW.continuity_probe(2000, 10, 4, 3)
    assert cont["changed_stable"] == 0 and cont["total"] == 2000 and cont["K_stable"] > 0


def test_standard_error_points():
    assert abs(SW.standard_error_points(450, 5000) - 0.4047) < 1e-3 and SW.standard_error_points(0, 5000) == 0.0 and SW.standard_error_points(1, 0) == 0.0


def test_coupling_example_has_four_alternatives_with_consistent_numbers():
    ex = SW.coupling_example(1000, 10, 4, 0.5)
    assert len(ex) == 4 and [e["Alternative"] for e in ex] == [1, 2, 3, 4]
    for e in ex:
        assert e["a_hoch"] >= e["a_tief"] and e["b_hoch"] >= e["b_tief"]
        assert e["gleichläufig"] >= e["gegenläufig"] - 1e-12                       # Σk = 0,5: c > 0
