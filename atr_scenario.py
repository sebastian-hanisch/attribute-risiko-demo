"""Der Kapazitätsfall mit zwei Zielen (Gewinn in GE, Servicegrad in %): ein **ausgedachtes Zahlenbeispiel**, ein Modell und kein Fall.

Zustände: Nachfrage hoch oder niedrig (je 1/2) und Lieferlage normal oder Engpass (je 1/2). Die Ränder sind fest; die Kopplung q = P(hoch und Engpass) steuert
die gemeinsame Verteilung (q = 1/4: unabhängig, q = 0: Engpass nur bei niedriger Nachfrage, q = 1/2: hohe Nachfrage immer mit Engpass).
Alternative G (groß bauen): Gewinn 200 bei hoher, −100 bei niedriger Nachfrage; Servicegrad 100 % bei normaler Lage, `s_g` im Engpass.
Alternative K (klein bauen oder mieten): Gewinn 30 sicher; Servicegrad 100 % bei normaler Lage, `s_k` im Engpass.
Einzelnutzen: Gewinn auf [−100, 200] normiert (exponentiell mit Risikotoleranz ρ in GE, ρ = ∞ linear), Servicegrad linear (u2 = Anteil).
Eingaben als `Fraction` rechnen exakt (nur bei ρ = ∞ für den Gewinn), Eingaben als `float` im Float; alle Formeln sind für beide gleich."""

import math
from dataclasses import dataclass
from fractions import Fraction as Fr

import numpy as np
from scipy import optimize

import atr_constants as C
import atr_utility as U

ALTS = ("G", "K")
FORMS = ("add", "mult")


@dataclass(frozen=True)
class Cell:
    demand: str    # "hoch" oder "niedrig"
    supply: str    # "normal" oder "Engpass"
    p: object      # Wahrscheinlichkeit der Zelle
    gain: int      # Gewinn in GE
    service: object  # Servicegrad als Anteil
    u1: object     # Einzelnutzen des Gewinns
    u2: object     # Einzelnutzen des Servicegrads (linear: gleich dem Anteil)


def cell_probabilities(q) -> dict:
    """Gemeinsame Verteilung bei gleichen Rändern (je 1/2): P(hoch, Engpass) = P(niedrig, normal) = q; 0 ≤ q ≤ 1/2."""
    if not 0 <= q <= Fr(1, 2):
        raise ValueError("q liegt außerhalb von 0 bis 1/2 (Grenzen der gemeinsamen Verteilung bei Rändern von je 1/2)")
    half = Fr(1, 2)
    return {("hoch", "normal"): half - q, ("hoch", "Engpass"): q, ("niedrig", "normal"): q, ("niedrig", "Engpass"): half - q}


def gain_of(alt: str, demand: str) -> int:
    if alt == "G":
        return C.GAIN_G_HIGH if demand == "hoch" else C.GAIN_G_LOW
    if alt == "K":
        return C.GAIN_K
    raise ValueError(alt)


def service_of(alt: str, supply: str, s_g, s_k):
    if supply == "normal":
        return Fr(1)
    return s_g if alt == "G" else s_k


def u1_of(gain: int, rho: float):
    """Einzelnutzen des Gewinns. ρ = ∞: exakter Bruch (gain + 100)/300; sonst float (exponentiell, normiert)."""
    if math.isinf(rho):
        return Fr(gain - C.GAIN_LO, C.GAIN_SPAN)
    return float(U.gain_utility(gain, rho, C.GAIN_LO, C.GAIN_HI))


def cells(alt: str, q, rho: float, s_g, s_k) -> list[Cell]:
    out = []
    for (demand, supply), p in cell_probabilities(q).items():
        g = gain_of(alt, demand)
        s = service_of(alt, supply, s_g, s_k)
        out.append(Cell(demand, supply, p, g, s, u1_of(g, rho), s))
    return out


def expected_utility(cs: list[Cell], k1, k2, form: str):
    """Erwartungsnutzen über die Zellen (aufgezählt): form = "mult" oder "add"."""
    f = U.u2_mult if form == "mult" else U.u2_add
    return sum(c.p * f(c.u1, c.u2, k1, k2) for c in cs)


def moments(cs: list[Cell]):
    """(E u1, E u2, Cov(u1, u2)) einer Alternative: die multiplikative Form ist EU = k1·E u1 + k2·E u2 + c·(E u1·E u2 + Cov)."""
    e1 = sum(c.p * c.u1 for c in cs)
    e2 = sum(c.p * c.u2 for c in cs)
    e12 = sum(c.p * c.u1 * c.u2 for c in cs)
    return e1, e2, e12 - e1 * e2


def eu(alt: str, q, rho: float, k1, k2, s_g, s_k, form: str):
    return expected_utility(cells(alt, q, rho, s_g, s_k), k1, k2, form)


def difference(q, rho: float, k1, k2, s_g, s_k, form: str):
    """EU(G) − EU(K): positiv heißt, dass G gewählt wird."""
    return eu("G", q, rho, k1, k2, s_g, s_k, form) - eu("K", q, rho, k1, k2, s_g, s_k, form)


def tipping_rho(q, k1, k2, s_g, s_k, form: str, grid=C.RHO_GRID) -> list[float]:
    """Risikotoleranzen ρ (GE) auf dem Raster, bei denen EU(G) − EU(K) das Vorzeichen wechselt (Nullstellen mit brentq zwischen Rasterpunkten).
    Leere Liste heißt: kein Wechsel zwischen den Rasterpunkten (nicht: für jedes ρ)."""
    q, k1, k2, s_g, s_k = float(q), float(k1), float(k2), float(s_g), float(s_k)
    f = lambda rho: float(difference(q, float(rho), k1, k2, s_g, s_k, form))  # noqa: E731
    vals = [f(r) for r in grid]
    out = []
    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            out.append(float(grid[i]))
        elif vals[i] * vals[i + 1] < 0:
            out.append(float(optimize.brentq(f, grid[i], grid[i + 1], xtol=1e-10)))
    return out


def tipping_q(rho: float, k1, k2, s_g, s_k):
    """Kopplung q*, bei der die multiplikative Form von G auf K wechselt. EU(G) − EU(K) ist in q linear (EU(K) hängt nicht von q ab), also genügen die Ränder q = 0
    und q = 1/2. None, wenn es in [0, 1/2] keinen Wechsel gibt. Mit Fraction-Eingaben (ρ = ∞) ist das Ergebnis ein exakter Bruch."""
    d0 = difference(Fr(0) if isinstance(k1, Fr) else 0.0, rho, k1, k2, s_g, s_k, "mult")
    d1 = difference(Fr(1, 2) if isinstance(k1, Fr) else 0.5, rho, k1, k2, s_g, s_k, "mult")
    if d0 == d1 or d0 * d1 > 0:
        return None
    return d0 / (d0 - d1) * (Fr(1, 2) if isinstance(d0, Fr) else 0.5)


def ce_gain_g(rho: float) -> float:
    """Sicherheitsäquivalent der Alternative G im Gewinn allein (Lotterie 200 / −100 je 1/2), in GE; −ρ·ln E exp(−x/ρ)."""
    return U.exp_certainty_equivalent([C.GAIN_G_HIGH, C.GAIN_G_LOW], [0.5, 0.5], rho)


def rho_star_single() -> float:
    """Kipp-Risikotoleranz des Gewinns allein: ρ mit u1(30; ρ) = 1/2, also Sicherheitsäquivalent von G gleich 30 GE (Wurzelsuche über die Einzelnutzen)."""
    f = lambda rho: float(U.gain_utility(C.GAIN_K, rho, C.GAIN_LO, C.GAIN_HI)) - 0.5  # noqa: E731
    return float(optimize.brentq(f, 50.0, 1e5, xtol=1e-12))


def curve(q, k1, k2, s_g, s_k, form: str, rhos) -> np.ndarray:
    """EU(G) − EU(K) über eine Liste von ρ-Werten (float), für das Diagramm."""
    q, k1, k2, s_g, s_k = float(q), float(k1), float(k2), float(s_g), float(s_k)
    return np.array([float(difference(q, float(r), k1, k2, s_g, s_k, form)) for r in rhos])
