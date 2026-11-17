"""Zufallsprobleme für die Häufigkeiten (Zähleinheit: Matrizen bzw. Probleme). Das sind **keine Fälle**, sondern Zufallsziehungen aus festen Generatoren.

* Form (Experiment 2): P Matrizen sicherer Ausgänge, n Alternativen, m Attribute, Einzelnutzen linear, gleichverteilt U(0,1) und unabhängig.
  Additiv mit w_i ∝ 1/(i + 1) (Summe 1), multiplikativ mit k_i = Σk·w_i. Gezählt: Matrizen, in denen Platz 1 verschieden ist.
* Risiko (Experiment 3): P Probleme, n = 10 Alternativen, m = 3 Attribute, S gleich wahrscheinliche Szenarien, Werte U(0,1) unabhängig. Verglichen wird der
  Erwartungsnutzen mit konkavem Einzelnutzen (ρ/Spanne) gegen dieselbe Form, auf die Attribut-Erwartungswerte angewandt (linear).
* Kopplung (Experiment 4): P Probleme, n Alternativen, zwei Attribute, je Alternative und Attribut ein Zwei-Punkt-Los (Werte U(0,1), je 1/2), Einzelnutzen linear,
  k = (0,6·Σk; 0,4·Σk). Dieselben Ränder, drei Kopplungen: gleichläufig, unabhängig, gegenläufig.
Zufall nur aus dem ganzzahligen SplitMix64 (`atr_rng`). Gleichstände (Abstand der beiden besten Werte unter 1e-12) zählen als Gleichstand, nicht als Wechsel."""

from dataclasses import dataclass

import numpy as np

import atr_constants as C
import atr_rng as RNG
import atr_utility as U

SALT_FORM, SALT_RISK, SALT_KOPPLUNG = 1, 2, 3
TAU_ROWS = 1000   # Kendall-Tau wird an den ersten höchstens 1000 Matrizen gerechnet (O(n²) je Matrix)


def draw(seed: int, salt: int, shape) -> np.ndarray:
    """Gleichverteilte Zahlen der Form `shape` aus dem Strom (Zufalls-Seed, Salz): ein eigener Strom je Experiment."""
    n = int(np.prod(shape))
    return RNG.uniform_array(int(seed) * 1000 + salt, n).reshape(shape)


def _gap(values: np.ndarray) -> np.ndarray:
    """Abstand zwischen bestem und zweitbestem Wert je Zeile (n ≥ 2); bei n = 1 gibt es keinen Abstand (unendlich)."""
    if values.shape[1] < 2:
        return np.full(values.shape[0], np.inf)
    s = np.partition(values, -2, axis=1)
    return s[:, -1] - s[:, -2]


def count_top_changes(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(Zahl der Zeilen mit anderem Platz 1, Zahl der Zeilen mit Gleichstand in a oder b). Zeilen = Probleme, Spalten = Alternativen."""
    tie = (_gap(a) < C.TIE) | (_gap(b) < C.TIE)
    changed = (np.argmax(a, axis=1) != np.argmax(b, axis=1)) & ~tie
    return int(changed.sum()), int(tie.sum())


def standard_error_points(count: int, total: int) -> float:
    """Standardfehler eines Anteils in Prozentpunkten."""
    if total <= 0:
        return 0.0
    p = count / total
    return 100.0 * float(np.sqrt(p * (1.0 - p) / total))


# --- Form (Experiment 2) ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FormRow:
    sum_k: float
    K: float
    changed: int
    ties: int
    total: int
    tau_mean: float
    tau_min: float


def form_values(P: int, n: int, m: int, seed: int) -> np.ndarray:
    return draw(seed, SALT_FORM, (P, n, m))


def form_row(u: np.ndarray, sum_k: float) -> FormRow:
    """Platz 1 der additiven Form (Gewichte w_i) gegen die der multiplikativen (k_i = Σk·w_i) auf den Matrizen u der Form (P, n, m)."""
    P, n, m = u.shape
    w = U.harmonic_weights(m)
    k = w * sum_k
    K = U.mult_constant(k)
    ua = U.add_utility(u, w)
    um = U.mult_utility(u, k, K)
    changed, ties = count_top_changes(ua, um)
    rows = min(P, TAU_ROWS)
    tau = U.kendall_tau(ua[:rows], um[:rows])
    return FormRow(float(sum_k), K, changed, ties, P, float(np.mean(tau)), float(np.min(tau)))


def form_table(P: int, n: int, m: int, seed: int, sums=C.SUM_K_VALUES) -> list[FormRow]:
    u = form_values(P, n, m, seed)
    return [form_row(u, s) for s in sums]


def example_matrix(P: int, n: int, m: int, seed: int, sum_k: float, top: int = 5):
    """Eine Beispielmatrix: die erste, deren Platz 1 sich zwischen den Formen ändert (sonst die erste überhaupt). Rückgabe: (Index, Änderung gefunden?,
    Liste der besten `top` Alternativen je Form als (Alternative, Nutzen))."""
    u = form_values(P, n, m, seed)
    w = U.harmonic_weights(m)
    k = w * sum_k
    ua, um = U.add_utility(u, w), U.mult_utility(u, k)
    differs = (np.argmax(ua, axis=1) != np.argmax(um, axis=1)) & ~((_gap(ua) < C.TIE) | (_gap(um) < C.TIE))
    idx = int(np.argmax(differs)) if differs.any() else 0
    best = lambda v: [(int(i), float(v[idx, i])) for i in np.argsort(-v[idx], kind="stable")[:top]]  # noqa: E731
    return idx, bool(differs.any()), best(ua), best(um)


def continuity_probe(P: int, n: int, m: int, seed: int, sum_k: float = 0.999999) -> dict:
    """Stetigkeitsprobe bei Σk nahe 1: stabiles K (Koeffizienten der Produktentwicklung) gegen das direkt gelöste K (Auslöschung)."""
    u = form_values(P, n, m, seed)
    w = U.harmonic_weights(m)
    k = w * sum_k
    K_ok, K_bad = U.mult_constant(k), U.mult_constant_naive(k)
    ua = U.add_utility(u, w)
    um_ok = U.mult_utility(u, k, K_ok)
    naive_ok = bool(np.isfinite(K_bad))
    um_bad = (np.prod(1.0 + K_bad * k * u, axis=-1) - 1.0) / K_bad if naive_ok else None   # direkt, wie die naive Rechnung es täte
    e2 = float(sum(k[i] * k[j] for i in range(m) for j in range(i + 1, m)))
    return {"K_stable": K_ok, "K_naive": K_bad, "K_series": (1.0 - float(k.sum())) / e2,
            "changed_stable": count_top_changes(ua, um_ok)[0], "changed_naive": count_top_changes(ua, um_bad)[0] if naive_ok else None, "total": P}


# --- Risiko (Experiment 3) -------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskRow:
    rho_norm: float
    changed: int
    ties: int
    total: int


def risk_values(P: int, S: int, seed: int, n: int = 10, m: int = C.RISK_M) -> np.ndarray:
    return draw(seed, SALT_RISK * 100 + S, (P, n, S, m))


def _risk_forms(form: str, m: int):
    w = U.harmonic_weights(m)
    if form == C.FORM_ADD:
        return lambda x: x @ w, w          # Erwartungswerte-Rechnung und Erwartungsnutzen teilen sich die Gewichte
    k = w * C.RISK_SUM_K
    K = U.mult_constant(k)
    return (lambda x: U.mult_utility(x, k, K)), k


def risk_row(X: np.ndarray, rho_norm: float, form: str = C.FORM_ADD) -> RiskRow:
    """Platz 1 nach dem Erwartungsnutzen (konkaver Einzelnutzen, Mittel über die Szenarien) gegen Platz 1 der Rechnung mit den Attribut-Erwartungswerten (linear).
    Beide mit derselben Form (additiv oder multiplikativ mit Σk = 0,5)."""
    P, n, S, m = X.shape
    form_fn, _ = _risk_forms(form, m)
    ref = form_fn(X.mean(axis=2))                              # (P, n): Form auf die Mittelwerte angewandt
    eu_ = form_fn(U.single_utility_exp(X, rho_norm)).mean(axis=2)  # (P, n): Erwartungsnutzen
    changed, ties = count_top_changes(ref, eu_)
    return RiskRow(float(rho_norm), changed, ties, P)


def risk_table(P: int, S: int, seed: int, form: str, rhos=C.RHO_NORM_OPTIONS) -> list[RiskRow]:
    X = risk_values(P, S, seed)
    return [risk_row(X, r, form) for r in rhos]


def risk_by_scenarios(P: int, seed: int, rho_norm: float, form: str, scenarios=(2, 3, 5, 10)) -> list[tuple[int, RiskRow]]:
    return [(S, risk_row(risk_values(P, S, seed), rho_norm, form)) for S in scenarios]


# --- Kopplung (Experiment 4) -----------------------------------------------------------------------------------------------------------------

COUPLINGS = ("gleichläufig", "unabhängig", "gegenläufig")
# Zellenwahrscheinlichkeiten je Kopplung über die vier Paare (Attribut 1 hoch/tief) × (Attribut 2 hoch/tief): hh, ht, th, tt; die Ränder sind in allen drei gleich (je 1/2)
COUPLING_CELLS = {
    "gleichläufig": (0.5, 0.0, 0.0, 0.5),
    "unabhängig": (0.25, 0.25, 0.25, 0.25),
    "gegenläufig": (0.0, 0.5, 0.5, 0.0),
}
K_SHARE = (0.6, 0.4)   # k = (0,6·Σk; 0,4·Σk)


@dataclass(frozen=True)
class CouplingResult:
    sum_k: float
    c: float
    total: int
    same_vs_opposite: tuple[int, int]     # (Wechsel, Gleichstände) des Platzes 1 bei der multiplikativen Form
    same_vs_independent: tuple[int, int]
    opposite_vs_independent: tuple[int, int]
    add_vs_mult_same: tuple[int, int]     # additiv gegen multiplikativ (gleichläufig)
    add_same_vs_opposite: tuple[int, int]  # additiv: gleichläufig gegen gegenläufig (muss 0 sein)
    diff_mean: float                      # EU(gleichläufig) − EU(gegenläufig) je Alternative (multiplikativ): Mittel, Minimum, Maximum
    diff_min: float
    diff_max: float


def coupling_values(P: int, n: int, seed: int) -> np.ndarray:
    return draw(seed, SALT_KOPPLUNG, (P, n, 2, 2))


def coupling_utilities(u: np.ndarray, sum_k: float) -> dict:
    """Erwartungsnutzen je Alternative bei den drei Kopplungen, jeweils additiv und multiplikativ, durch **Aufzählung der vier Zellen** (hoch/tief × hoch/tief).
    u: (P, n, 2 Ausgänge, 2 Attribute) mit den Werten (= Einzelnutzen, linear). Rückgabe: {Kopplung: {"mult": (P, n), "add": (P, n)}} und "c"."""
    a, b = u[..., 0], u[..., 1]
    ah, at_ = a.max(axis=2), a.min(axis=2)
    bh, bt = b.max(axis=2), b.min(axis=2)
    k1, k2 = K_SHARE[0] * sum_k, K_SHARE[1] * sum_k
    c = 1.0 - k1 - k2
    if abs(c) < U.ZERO_SUM_TOL:
        c = 0.0
    pairs = ((ah, bh), (ah, bt), (at_, bh), (at_, bt))
    out = {"c": c}
    for name, probs in COUPLING_CELLS.items():
        mult = sum(p * (k1 * x + k2 * y + c * x * y) for p, (x, y) in zip(probs, pairs))
        add = sum(p * (k1 * x + k2 * y) / (k1 + k2) for p, (x, y) in zip(probs, pairs))
        out[name] = {"mult": mult, "add": add}
    return out


def coupling_result(u: np.ndarray, sum_k: float) -> CouplingResult:
    eu_ = coupling_utilities(u, sum_k)
    g, x, ind = (eu_[n]["mult"] for n in ("gleichläufig", "gegenläufig", "unabhängig"))
    diff = g - x
    return CouplingResult(
        float(sum_k), float(eu_["c"]), u.shape[0],
        count_top_changes(g, x), count_top_changes(g, ind), count_top_changes(x, ind),
        count_top_changes(eu_["gleichläufig"]["add"], g),
        count_top_changes(eu_["gleichläufig"]["add"], eu_["gegenläufig"]["add"]),
        float(diff.mean()), float(diff.min()), float(diff.max()),
    )


def coupling_table(P: int, n: int, seed: int, sums=C.SUM_K_VALUES) -> list[CouplingResult]:
    u = coupling_values(P, n, seed)
    return [coupling_result(u, s) for s in sums]


def coupling_example(P: int, n: int, seed: int, sum_k: float, rows: int = 4) -> list[dict]:
    """Die ersten `rows` Alternativen des ersten Problems: Randwerte und Erwartungsnutzen bei den drei Kopplungen (multiplikativ) und additiv."""
    u = coupling_values(P, n, seed)
    eu_ = coupling_utilities(u[:1], sum_k)
    out = []
    for i in range(min(rows, n)):
        a, b = u[0, i, :, 0], u[0, i, :, 1]
        out.append({"Alternative": i + 1, "a_hoch": float(a.max()), "a_tief": float(a.min()), "b_hoch": float(b.max()), "b_tief": float(b.min()),
                    "additiv": float(eu_["gleichläufig"]["add"][0, i]), **{name: float(eu_[name]["mult"][0, i]) for name in COUPLINGS}})
    return out
