"""Auswertungen der Abschnitte der App und ihre Verdikte (reine Funktionen: (Art, Text)).

Jede Auswertung ruft nur die Einheiten `atr_scenario`, `atr_sweeps` und `atr_utility` auf; die App legt sie hinter `st.cache_data`. Alle Zahlen in den Verdikten
kommen aus der Rechnung dieses Moduls, nicht aus der README. Der Kapazitätsfall ist ein ausgedachtes Zahlenbeispiel (Modell, kein Fall)."""

import math
from dataclasses import dataclass
from fractions import Fraction

import atr_constants as C
import atr_scenario as SC
import atr_sweeps as SW
import atr_utility as U

EPS = 1e-12


# --- Formatierung (Dezimalkomma, Minuszeichen) -----------------------------------------------------------------------------------------------

def de(x: float, nd: int = 2, sign: bool = False) -> str:
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def _frac_text(x: Fraction) -> str:
    return str(x).replace("-", "−")


def fr(x, nd: int = 4) -> str:
    """Bruch mit Dezimalwert (23/40 → „23/40 (0,5750)“); Gleitkommazahlen nur als Dezimalzahl."""
    if isinstance(x, Fraction):
        return _frac_text(x) if x.denominator == 1 else f"{_frac_text(x)} ({de(float(x), nd)})"
    return de(float(x), nd)


def fr_short(x) -> str:
    return _frac_text(x) if isinstance(x, Fraction) else de(float(x), 4)


def pct(count: int, total: int, nd: int = 1) -> str:
    return de(100.0 * count / total, nd) if total else de(0.0, nd)


def winner(d) -> str:
    return "G" if d > EPS else ("K" if d < -EPS else "keine (Gleichstand)")


def ge_list(values: list[float]) -> str:
    if not values:
        return "kein Wechsel im Raster"
    return " und ".join(de(v, 1) for v in values) + " GE"


# --- Kapazitätsfall (Experiment 1 und 3b) ----------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class HandSettings:
    k1: Fraction = U.as_fraction(C.DEFAULT_K1)
    k2: Fraction = U.as_fraction(C.DEFAULT_K2)
    q: Fraction = U.as_fraction(C.DEFAULT_Q)
    rho: float = C.DEFAULT_RHO
    s_g: Fraction = Fraction(C.DEFAULT_SERVICE_G, 100)
    s_k: Fraction = Fraction(C.DEFAULT_SERVICE_K, 100)


@dataclass(frozen=True)
class HandResult:
    hs: HandSettings
    exact: bool
    c: Fraction
    K: object                      # K = c/(k1·k2) (Fraction), bei Σk = 1: 0
    cells: dict                    # {"G": [Cell], "K": [Cell]}
    eu: dict                       # {(Alternative, Form): Erwartungsnutzen}
    moments: dict                  # {Alternative: (E u1, E u2, Cov)}
    d_mult: object
    d_add: object
    q_star: object                 # Kipp-Kopplung der multiplikativen Form (None: kein Wechsel in 0 … ½)
    slope_G: object                # Steigung von EU(G) (multiplikativ) in q
    kipp: dict                     # {"add": [ρ], "mult0": [ρ], "mult25": [ρ], "mult50": [ρ]} auf dem Raster


def analyse_hand(hs: HandSettings) -> HandResult:
    exact = math.isinf(hs.rho)
    # Rechnungen im Gleitkomma, wenn ρ endlich ist; die Brüche bleiben nur bei ρ = ∞ im Spiel
    k1, k2, q, s_g, s_k = (hs.k1, hs.k2, hs.q, hs.s_g, hs.s_k) if exact else (float(hs.k1), float(hs.k2), float(hs.q), float(hs.s_g), float(hs.s_k))
    cs = {a: SC.cells(a, q, hs.rho, s_g, s_k) for a in SC.ALTS}
    eu = {(a, f): SC.expected_utility(cs[a], k1, k2, f) for a in SC.ALTS for f in SC.FORMS}
    mom = {a: SC.moments(cs[a]) for a in SC.ALTS}
    c = U.cross_coefficient(hs.k1, hs.k2)
    K = c / (hs.k1 * hs.k2)
    zero, half = (Fraction(0), Fraction(1, 2)) if exact else (0.0, 0.5)
    g0 = SC.eu("G", zero, hs.rho, k1, k2, s_g, s_k, "mult")
    g1 = SC.eu("G", half, hs.rho, k1, k2, s_g, s_k, "mult")
    kipp = {"add": SC.tipping_rho(q, k1, k2, s_g, s_k, "add"),
            "mult0": SC.tipping_rho(0.0, k1, k2, s_g, s_k, "mult"),
            "mult25": SC.tipping_rho(0.25, k1, k2, s_g, s_k, "mult"),
            "mult50": SC.tipping_rho(0.5, k1, k2, s_g, s_k, "mult")}
    return HandResult(hs, exact, c, K, cs, eu, mom, eu[("G", "mult")] - eu[("K", "mult")], eu[("G", "add")] - eu[("K", "add")],
                      SC.tipping_q(hs.rho, k1, k2, s_g, s_k), (g1 - g0) / half, kipp)


def verdict_hand(r: HandResult) -> tuple[str, str]:
    """Verdikt des Kapazitätsfalls (Experiment 1): welche Alternative wählt welche Form, und was die Kopplung ändert."""
    hs = r.hs
    sk = hs.k1 + hs.k2
    head = f"k = ({fr_short(hs.k1)}; {fr_short(hs.k2)}), Σk = {fr_short(sk)}, ρ = {'∞' if r.exact else de(hs.rho, 1) + ' GE'}, q = {fr_short(hs.q)}. "
    tail = " Modell mit ausgedachten Servicegraden, kein Fall."
    if r.c == 0:
        return "success", (head + f"Σk = 1, also c = 0 und K = 0: die multiplikative Form ist die additive. Beide wählen {winner(r.d_add)} (G − K = {fr(r.d_add)}), bei jeder Kopplung q." + tail)
    text = (head + f"Multiplikativ: EU(G) = {fr(r.eu[('G', 'mult')])}, EU(K) = {fr(r.eu[('K', 'mult')])}, wählt {winner(r.d_mult)}. "
            f"Additiv: EU(G) = {fr(r.eu[('G', 'add')])}, EU(K) = {fr(r.eu[('K', 'add')])}, wählt {winner(r.d_add)}. ")
    kind = "info"
    if winner(r.d_mult) != winner(r.d_add):
        text += "**Die Form entscheidet.** "
        kind = "warning"
    if r.q_star is not None:
        text += f"Die multiplikative Form wechselt bei q* = {fr(r.q_star)} von G auf K (Steigung von EU(G) in q: {fr_short(r.slope_G)}); die additive hängt nicht von q ab."
    else:
        text += "Die multiplikative Form wechselt für kein q zwischen 0 und ½ ihre Wahl; die additive hängt nicht von q ab."
    return kind, text + tail


def verdict_kipp(r: HandResult) -> tuple[str, str]:
    """Verdikt zu den Kipp-Risikotoleranzen (Experiment 3, Kapazitätsfall)."""
    k = r.kipp
    return "info", (f"Im Gewinn allein kippt die Wahl bei ρ* = {de(C.RHO_STAR_SINGLE, 1)} GE. Mit dem zweiten Attribut (k = ({fr_short(r.hs.k1)}; {fr_short(r.hs.k2)}), "
                    f"Servicegrad im Engpass G {de(100 * float(r.hs.s_g), 0)} %, K {de(100 * float(r.hs.s_k), 0)} %) kippt sie additiv bei {ge_list(k['add'])} (für jedes q), "
                    f"multiplikativ bei {ge_list(k['mult0'])} (q = 0), bei {ge_list(k['mult25'])} (q = ¼) und bei q = ½: {ge_list(k['mult50'])}. "
                    "Raster 5 GE bis 10⁹ GE (17 Punkte); „kein Wechsel im Raster“ heißt nicht „für jedes ρ“. ρ* ist hier keine Eigenschaft des Gewinns mehr.")


# --- Zufallsexperimente ----------------------------------------------------------------------------------------------------------------------

def verdict_form(row: SW.FormRow, n: int, m: int) -> tuple[str, str]:
    """Verdikt Experiment 2 (Form bei sicheren Ausgängen) für die gewählte Σk."""
    base = f"{row.total} Matrizen, n = {n} Alternativen, m = {m} Attribute (lineare, gleichverteilte Einzelnutzen). "
    if abs(row.sum_k - 1.0) < 1e-9:
        return "success", (base + f"Σk = 1 (K = 0): Platz 1 ist in {row.changed} von {row.total} Matrizen verschieden ({row.ties} Gleichstände): die multiplikative Form ist die additive.")
    se = SW.standard_error_points(row.changed, row.total)
    same = row.total - row.changed - row.ties
    return "info", (base + f"Σk = {de(row.sum_k, 1)} (K = {de(row.K, 2, True)}): Platz 1 ist in {row.changed} von {row.total} Matrizen verschieden ({pct(row.changed, row.total)} %, "
                    f"Standardfehler {de(se, 1)} Punkte); in {same} Matrizen ist die additive Näherung für Platz 1 unauffällig. Rangfolge insgesamt: Kendall-Tau im Mittel "
                    f"{de(row.tau_mean, 3)}, kleinster Wert {de(row.tau_min, 3)} (erste {min(row.total, SW.TAU_ROWS)} Matrizen).")


def sci(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma (2.61e-06 → 2,610e-06)."""
    return f"{x:.3e}".replace(".", ",")


def text_continuity(cont: dict) -> str:
    """Stetigkeitsprobe bei Σk = 0,999999: stabiles K gegen das direkt gelöste K (Rechenfalle)."""
    head = (f"**Stetigkeitsprobe bei Σk = 0,999999:** K aus den Koeffizienten der Produktentwicklung = {sci(cont['K_stable'])} (Näherung (1 − Σk)/e₂ = {sci(cont['K_series'])}); "
            f"{cont['changed_stable']} von {cont['total']} Matrizen mit anderem Platz 1. Löst man die Gleichung direkt, löscht sich fast alles aus: ")
    if cont["changed_naive"] is None:
        return head + "die Wurzelsuche findet gar kein Vorzeichenintervall (kein Ergebnis, ein Rechenfehler)."
    return head + f"K = {sci(cont['K_naive'])}, scheinbar {cont['changed_naive']} von {cont['total']} Matrizen mit anderem Platz 1 (kein Befund, ein Rechenfehler)."


def verdict_risk(row: SW.RiskRow, S: int, form: str) -> tuple[str, str]:
    """Verdikt Experiment 3a: Mittelwerte einsetzen statt Erwartungsnutzen."""
    se = SW.standard_error_points(row.changed, row.total)
    text = (f"{row.total} Probleme, 10 Alternativen, 3 Attribute, {S} Szenarien, Form {form}, ρ/Spanne = {de(row.rho_norm, 2)}: Platz 1 ändert sich in {row.changed} von {row.total} "
            f"Problemen ({pct(row.changed, row.total)} %, Standardfehler {de(se, 1)} Punkte), wenn man die Attribut-Erwartungswerte einsetzt statt den Erwartungsnutzen zu rechnen. "
            f"Der Kapazitätsfall liegt bei ρ/Spanne = {de(C.RHO_STAR_SINGLE / C.GAIN_SPAN, 2)}.")
    if form != C.FORM_ADD:
        text += " In der multiplikativen Form steckt in dem Wert auch die Kopplung der Szenarien (Erwartung eines Produkts gegen Produkt der Erwartungen)."
    return "info", text


def verdict_scenarios(rows: list[tuple[int, SW.RiskRow]]) -> tuple[str, str]:
    first, last = rows[0], rows[-1]
    return "info", (f"Szenarienzahl S = {first[0]}: {pct(first[1].changed, first[1].total)} %, S = {last[0]}: {pct(last[1].changed, last[1].total)} % der Probleme "
                    f"(je {first[1].total} Probleme, ρ/Spanne = {de(first[1].rho_norm, 2)}). Mit mehr Szenarien sinkt der Anteil in diesen Zufallsproblemen "
                    f"{'nicht' if last[1].changed >= first[1].changed - 0.01 * first[1].total else 'doch'}; die Ursache ist nicht untersucht.")


def verdict_coupling(r: SW.CouplingResult, n: int) -> tuple[str, str]:
    """Verdikt Experiment 4: gleiche Ränder, andere Kopplung."""
    base = f"{r.total} Probleme, {n} Alternativen, zwei Zwei-Punkt-Lose je Alternative. "
    add_txt = f"Additiv ändert sich Platz 1 mit der Kopplung in {r.add_same_vs_opposite[0]} von {r.total} Problemen. "
    if abs(r.c) < 1e-9:
        return "success", (base + add_txt + f"Σk = 1 (c = 0): auch multiplikativ in {r.same_vs_opposite[0]} von {r.total} (gleichläufig gegen gegenläufig).")
    sign = "gleichläufig vor gegenläufig" if r.c > 0 else "gegenläufig vor gleichläufig"
    ok_sign = (r.diff_min >= -1e-12) if r.c > 0 else (r.diff_max <= 1e-12)
    return "info", (base + add_txt + f"Multiplikativ (Σk = {de(r.sum_k, 1)}, c = {de(r.c, 1, True)}): gleichläufig gegen gegenläufig wechselt Platz 1 in "
                    f"{r.same_vs_opposite[0]} von {r.total} Problemen ({pct(r.same_vs_opposite[0], r.total)} %), gleichläufig gegen unabhängig in {r.same_vs_independent[0]}, "
                    f"gegenläufig gegen unabhängig in {r.opposite_vs_independent[0]}. Der Erwartungsnutzen (gleichläufig minus gegenläufig) liegt je Alternative zwischen "
                    f"{de(r.diff_min, 3, True)} und {de(r.diff_max, 3, True)}"
                    f"{'; das Vorzeichen von c entscheidet, welche Kopplung vorn liegt: ' + sign if ok_sign else ''}.")


# --- Kernfragen (kurze Fassungen, live) ------------------------------------------------------------------------------------------------------

def kern_one(r: HandResult) -> tuple[str, str]:
    sk = r.hs.k1 + r.hs.k2
    if r.c == 0:
        return "info", f"Σk = {fr_short(sk)}: c = 0, die Form ist additiv, ein Kreuzterm fehlt."
    sign = "positiv, der Kreuzterm belohnt Ausgewogenheit" if r.c > 0 else "negativ, der Kreuzterm belohnt Einseitigkeit"
    return "info", f"Σk = {fr_short(sk)}: c = 1 − Σk = {fr_short(r.c)} ist {sign}; K = {fr_short(r.K)}."


def kern_two(form_row: SW.FormRow, coupling: SW.CouplingResult) -> tuple[str, str]:
    return "info", (f"Form (sichere Ausgänge, Σk = {de(form_row.sum_k, 1)}): Platz 1 verschieden in {form_row.changed} von {form_row.total} Matrizen. "
                    f"Kopplung (Σk = {de(coupling.sum_k, 1)}): gleichläufig gegen gegenläufig in {coupling.same_vs_opposite[0]} von {coupling.total} Problemen, additiv in {coupling.add_same_vs_opposite[0]}.")


def kern_three(row: SW.RiskRow, form: str) -> tuple[str, str]:
    return "info", (f"Form {form}, ρ/Spanne = {de(row.rho_norm, 2)}: Platz 1 ändert sich in {row.changed} von {row.total} Problemen ({pct(row.changed, row.total)} %), "
                    "wenn die Attribut-Erwartungswerte statt des Erwartungsnutzens eingesetzt werden.")
