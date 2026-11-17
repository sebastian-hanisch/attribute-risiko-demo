"""Gemeinsame Fixtures: das Handbeispiel in Brüchen (Kapazitätsfall mit Servicegrad, k = (2/5, 3/10)), an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import math
import sys
from fractions import Fraction as Fr
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def hand():
    """k1 = 2/5, k2 = 3/10, Servicegrad im Engpass G 1/2 und K 3/5, ρ = ∞ (alles in Brüchen). Handrechnung siehe tests/test_scenario.py."""
    return {"k1": Fr(2, 5), "k2": Fr(3, 10), "s_g": Fr(1, 2), "s_k": Fr(3, 5), "rho": math.inf}
