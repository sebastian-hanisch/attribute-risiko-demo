"""Zufallsgenerator: dieselbe Folge in Klasse und Feld, bekannte Referenzwerte, Gleichverteilung, Plattformfestigkeit (nur Ganzzahlarithmetik)."""

import numpy as np

import atr_rng as R


def test_array_stream_equals_class_stream():
    g = R.SplitMix64(12345)
    assert [g.next_u64() for _ in range(50)] == [int(v) for v in R.splitmix_array(12345, 50)]
    g = R.SplitMix64(7)
    assert [g.random() for _ in range(50)] == list(R.uniform_array(7, 50))
    assert [g.random() for _ in range(5)] == list(R.uniform_array(7, 55)[50:])           # Fortsetzung der Folge


def test_known_splitmix_values_for_seed_zero():
    """Referenzfolge von SplitMix64 mit Zustand 0 (Vigna): 0xe220a8397b1dcdaf, 0x6e789e6aa1b965f4."""
    g = R.SplitMix64(0)
    assert g.next_u64() == 0xE220A8397B1DCDAF
    assert g.next_u64() == 0x6E789E6AA1B965F4
    assert [int(v) for v in R.splitmix_array(0, 2)] == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4]


def test_uniform_in_unit_interval_with_correct_moments():
    u = R.uniform_array(99, 200000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 5 * (1 / np.sqrt(12 * 200000))                          # 5 Standardfehler
    assert abs(u.var() - 1 / 12) < 0.002


def test_seed_beyond_64_bit_is_masked_and_streams_differ():
    assert list(R.uniform_array((1 << 64) + 5, 3)) == list(R.uniform_array(5, 3))
    assert list(R.uniform_array(1, 5)) != list(R.uniform_array(2, 5))
