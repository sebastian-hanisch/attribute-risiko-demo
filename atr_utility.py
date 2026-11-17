"""Einzelnutzen, Skalierungskonstante K, multiplikative und additive Form (nur numpy, scipy.optimize.brentq und Standardbibliothek).

Konvention (vor der ersten Gleichung):
* Einzelnutzen u_i(x_i) in [0, 1], normiert auf die Spanne des Attributs (schlechtester Wert 0, bester 1).
* Skalierungskonstanten k_i (nicht „Gewichte“): k_i = U(Attribut i am besten, alle übrigen am schlechtesten).
* Multiplikative Form: 1 + K·U = Produkt_i (1 + K·k_i·u_i) mit 1 + K = Produkt_i (1 + K·k_i); K = 0 (additiv) genau bei Σk = 1.
  Für zwei Attribute ausmultipliziert: U = k1·u1 + k2·u2 + c·u1·u2 mit c = 1 − k1 − k2 = K·k1·k2.
* Additive Form: U = Σ (k_i / Σk) · u_i.
Alles hier ist eigene Nachrechnung in diesem Modell (Lehrbuchformen, keine Neuheit)."""

import math
from fractions import Fraction

import numpy as np
from scipy import optimize

ZERO_SUM_TOL = 1e-12   # |Σk − 1| darunter gilt K = 0


def harmonic_weights(m: int) -> np.ndarray:
    """Gewichte w_i ∝ 1/(i + 1), i = 1 … m, Summe 1 (für m = 2: 0,6 und 0,4). Eine feste Regel statt freier Gewichte."""
    if m < 1:
        raise ValueError("m muss mindestens 1 sein")
    w = 1.0 / (np.arange(1, m + 1) + 1.0)
    return w / w.sum()


def _check_k(k) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    if k.ndim != 1 or k.size < 1:
        raise ValueError("k muss ein nichtleerer Vektor sein")
    if np.any(k < 0) or np.any(k >= 1):
        raise ValueError("jedes k_i muss in [0, 1) liegen")
    return k


def _elementary(k: np.ndarray) -> np.ndarray:
    """Koeffizienten e_0 … e_m der Produktentwicklung Produkt (1 + K·k_i) = Σ e_j K^j (e_j = j-te elementarsymmetrische Funktion der k_i)."""
    coef = np.array([1.0])
    for k_i in k:
        coef = np.convolve(coef, [k_i, 1.0])   # nach fallender Potenz von K
    return coef[::-1]


def mult_constant(k) -> float:
    """K aus 1 + K = Produkt (1 + K·k_i), K ≠ 0 (K = 0 bei Σk = 1). Gelöst mit den Koeffizienten der Produktentwicklung:
    g(K) = (Produkt − 1 − K)/K = (e_1 − 1) + K·(e_2 + e_3·K + …). Die direkte Auswertung von (Produkt − 1 − K)/K löscht sich bei Σk nahe 1 aus (siehe `mult_constant_naive`)."""
    k = _check_k(k)
    s = float(k.sum())
    if abs(s - 1.0) < ZERO_SUM_TOL:
        return 0.0
    if int(np.count_nonzero(k)) < 2:
        raise ValueError("mit weniger als zwei positiven k_i gibt es für Σk ≠ 1 keine multiplikative Form")
    e = _elementary(k)
    higher = e[:1:-1] if len(e) > 2 else np.array([0.0])   # e_m, …, e_2
    g = lambda K: (e[1] - 1.0) + K * np.polyval(higher, K)  # noqa: E731
    if s < 1.0:
        hi = 1.0
        while g(hi) <= 0.0:
            hi *= 2.0
            if hi > 1e300:
                raise ValueError("kein Wurzelintervall für K gefunden")
        return float(optimize.brentq(g, 1e-13, hi, xtol=1e-18, rtol=1e-15, maxiter=2000))
    return float(optimize.brentq(g, -1.0 + 1e-12, -1e-13, xtol=1e-18, rtol=1e-15, maxiter=2000))


def mult_constant_naive(k) -> float:
    """Nur zur Vorführung der Rechenfalle: g(K) = (Produkt(1 + K·k_i) − 1 − K)/K direkt ausgewertet, Wurzelsuche ab 1e-12. Bei Σk nahe 1 liefert das ein falsches K (oder NaN, wenn die Wurzelsuche scheitert)."""
    k = _check_k(k)
    g = lambda K: (np.prod(1.0 + K * k) - 1.0 - K) / K  # noqa: E731
    try:
        return float(optimize.brentq(g, 1e-12, 1e6, xtol=1e-16, rtol=1e-15, maxiter=1000))
    except ValueError:   # die Auslöschung kann auch jedes Vorzeichenintervall zerstören: dann gibt es gar kein Ergebnis
        return float("nan")


def mult_utility(u, k, K: float | None = None):
    """Multiplikativer Nutzen; u hat die Form (…, m) mit Einzelnutzen in [0, 1]. Auswertung per expm1/log1p, stabil auch bei sehr kleinem K."""
    k = _check_k(k)
    u = np.asarray(u, dtype=float)
    if K is None:
        K = mult_constant(k)
    if K == 0.0:
        return u @ k
    return np.expm1(np.sum(np.log1p(K * k * u), axis=-1)) / K


def add_utility(u, k):
    """Additiver Nutzen mit den auf Summe 1 normierten Konstanten."""
    k = np.asarray(k, dtype=float)
    return np.asarray(u, dtype=float) @ (k / k.sum())


def u2_mult(u1, u2, k1, k2):
    """Zwei Attribute, ausmultipliziert: k1·u1 + k2·u2 + (1 − k1 − k2)·u1·u2. Gilt für Fraction (exakt) und float."""
    return k1 * u1 + k2 * u2 + (1 - k1 - k2) * u1 * u2


def u2_add(u1, u2, k1, k2):
    return (k1 * u1 + k2 * u2) / (k1 + k2)


def cross_coefficient(k1, k2):
    """c = 1 − k1 − k2 (= K·k1·k2): positiv bei Σk < 1, negativ bei Σk > 1, 0 bei Σk = 1."""
    return 1 - k1 - k2


def single_utility_exp(x01, rho_norm: float):
    """Einzelnutzen auf [0, 1], normiert: (1 − exp(−x/ρ)) / (1 − exp(−1/ρ)); ρ = ∞ ist linear. x01 liegt im Intervall [0, 1] (Anteil der Spanne)."""
    x01 = np.asarray(x01, dtype=float)
    if rho_norm <= 0:
        raise ValueError("ρ muss positiv sein")
    if math.isinf(rho_norm):
        return x01
    return np.expm1(-x01 / rho_norm) / np.expm1(-1.0 / rho_norm)


def gain_utility(gain_ge, rho_ge: float, lo: float, hi: float):
    """Einzelnutzen eines Gewinns in GE: auf die Spanne [lo, hi] normiert, ρ in GE. Eine Spanne 0 hat keinen Einzelnutzen (nicht definierbar): ValueError."""
    if not hi > lo:
        raise ValueError("ein Attribut mit Spanne 0 hat keinen normierten Einzelnutzen")
    return single_utility_exp((np.asarray(gain_ge, dtype=float) - lo) / (hi - lo), rho_ge / (hi - lo))


def exp_certainty_equivalent(values, probs, rho: float) -> float:
    """Sicherheitsäquivalent −ρ·ln E[exp(−x/ρ)] einer Lotterie (CARA, ρ = Risikotoleranz in Einheiten des Gewinns). Anderer Rechenweg als der normierte Einzelnutzen."""
    values = np.asarray(values, dtype=float)
    probs = np.asarray(probs, dtype=float)
    shifted = values - values.min()   # gegen Überlauf: ln E exp(−x/ρ) = −min/ρ + ln E exp(−(x − min)/ρ)
    return float(values.min() - rho * math.log(float(np.sum(probs * np.exp(-shifted / rho)))))


def kendall_tau(a, b) -> np.ndarray:
    """Kendall-Tau je Zeile für Felder (P, n) ohne Gleichstände (tau-a, über alle geordneten Paare). n < 2: NaN."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = a.shape[1]
    if n < 2:
        return np.full(a.shape[0], np.nan)
    da = np.sign(a[:, :, None] - a[:, None, :])
    db = np.sign(b[:, :, None] - b[:, None, :])
    return (da * db).sum(axis=(1, 2)) / float(n * (n - 1))


def as_fraction(x: float, denominator: int = 100) -> Fraction:
    """Regler-Wert (float) als Bruch mit dem gegebenen Nenner (0,35 → 7/20)."""
    return Fraction(int(round(float(x) * denominator)), denominator)
