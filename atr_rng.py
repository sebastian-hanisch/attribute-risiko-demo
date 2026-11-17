"""Ganzzahliger Zufallsgenerator (SplitMix64): gleiche Zufalls-Seeds geben auf jeder Plattform dieselbe Folge.

Herkunft: die Klasse `SplitMix64` ist die Kopie der geprüften Vorlage `tools/templates/rng_splitmix64_exact.py` des Website-Repos, mit Präfix umbenannt und ohne
die dort nur für Normalzahlen gebrauchten Funktionen `exact_log` und `exact_cos` (diese Demo zieht nur gleichverteilte Zahlen, `random()` ist reine Ganzzahlarithmetik).
`splitmix_array` und `uniform_array` rechnen dieselbe Folge vektorisiert (der Zustand ist ein Zähler, die Ausgabe eine Mischfunktion des Zählers; Muster aus
`lager-chaos-demo`); `tests/test_rng.py` prüft die Gleichheit mit der Klasse. Python-`hash()`, `random` und `numpy.random` sind für feste Zufalls-Seeds über
Plattformen und Versionen nicht geeignet."""

import numpy as np

MASK = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15
TWO53 = float(1 << 53)


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & MASK

    def next_u64(self) -> int:
        self.state = (self.state + GAMMA) & MASK
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def random(self) -> float:
        """Gleichverteilt in [0, 1) mit 53 Bit."""
        return (self.next_u64() >> 11) / TWO53


def splitmix_array(seed: int, n: int) -> np.ndarray:
    """Die ersten n Ausgaben von SplitMix64(seed) als uint64-Feld (Zähler plus Mischfunktion, vektorisiert)."""
    with np.errstate(over="ignore"):
        idx = np.arange(1, n + 1, dtype=np.uint64)
        z = np.uint64(seed & MASK) + idx * np.uint64(GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def uniform_array(seed: int, n: int) -> np.ndarray:
    """n gleichverteilte Zahlen in [0, 1): dieselbe Folge wie `SplitMix64(seed).random()`."""
    return (splitmix_array(seed, n) >> np.uint64(11)).astype(np.float64) / TWO53
