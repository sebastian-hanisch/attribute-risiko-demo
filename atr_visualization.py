"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""


import numpy as np
import plotly.graph_objects as go

import atr_constants as C
import atr_scenario as SC
import atr_sweeps as SW


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_eu_bars(res):
    """Erwartungsnutzen je Alternative und Form; bei der multiplikativen Form getrennt in Randterm k1·E u1 + k2·E u2 und Kreuzterm c·E[u1·u2]."""
    k1, k2 = float(res.hs.k1), float(res.hs.k2)
    c = float(res.c)
    xs, edge, cross = [], [], []
    for alt in SC.ALTS:
        e1, e2, cov = res.moments[alt]
        e12 = float(e1) * float(e2) + float(cov)
        xs.append(f"{alt}, additiv")
        edge.append(float(res.eu[(alt, "add")]))
        cross.append(0.0)
        xs.append(f"{alt}, multiplikativ")
        edge.append(k1 * float(e1) + k2 * float(e2))
        cross.append(c * e12)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=xs, y=edge, name="Randterm (k₁·E u₁ + k₂·E u₂; additiv: normiert)", marker_color=C.COLOR_ADD, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Bar(x=xs, y=cross, name="Kreuzterm c·E[u₁·u₂]", marker_color=C.COLOR_CROSS, hovertemplate="%{x}: %{y:.4f}<extra></extra>"))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="Erwartungsnutzen")
    return _base(fig)


def build_rho_curves(res):
    """EU(G) − EU(K) über die Risikotoleranz ρ (GE, logarithmisch): additiv und multiplikativ bei q = 0, ¼, ½; Kippstellen als Marker."""
    hs = res.hs
    rhos = np.geomspace(20.0, 1e5, 70)
    series = [("additiv (jedes q)", "add", 0.0, C.COLOR_ADD, res.kipp["add"]), ("multiplikativ, q = 0", "mult", 0.0, "#e0a030", res.kipp["mult0"]),
              ("multiplikativ, q = ¼", "mult", 0.25, C.COLOR_MULT, res.kipp["mult25"]), ("multiplikativ, q = ½", "mult", 0.5, "#7b4aa0", res.kipp["mult50"])]
    fig = go.Figure()
    for name, form, q, color, tips in series:
        ys = SC.curve(q, hs.k1, hs.k2, hs.s_g, hs.s_k, form, rhos)
        fig.add_trace(go.Scatter(x=rhos, y=ys, mode="lines", name=name, line=dict(color=color, width=2.5), hovertemplate="ρ = %{x:.0f} GE: %{y:.4f}<extra></extra>"))
        if tips:
            fig.add_trace(go.Scatter(x=tips, y=[0.0] * len(tips), mode="markers", marker=dict(color=color, size=10, line=dict(color="white", width=1)),
                                     showlegend=False, hovertemplate="Kipp bei ρ = %{x:.1f} GE<extra></extra>"))
    fig.add_hline(y=0.0, line_color=C.COLOR_GRID)
    fig.add_vline(x=C.RHO_STAR_SINGLE, line_dash="dot", line_color="rgba(120,120,120,0.8)", annotation_text="ρ* = 555,8 (Gewinn allein)", annotation_position="top")
    fig.update_xaxes(title_text="Risikotoleranz ρ des Gewinns (GE, ↑ weniger risikoscheu)", type="log")
    fig.update_yaxes(title_text="EU(G) − EU(K), positiv: G")
    return _base(fig, 400)


def build_form_bars(rows):
    """Anteil Platz 1 verschieden je Σk mit Standardfehler."""
    xs = [f"Σk = {r.sum_k:g}".replace(".", ",") for r in rows]
    ys = [100.0 * r.changed / r.total for r in rows]
    err = [SW.standard_error_points(r.changed, r.total) for r in rows]
    fig = go.Figure(go.Bar(x=xs, y=ys, error_y=dict(type="data", array=err), marker_color=C.COLOR_MULT, hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_yaxes(title_text="Platz 1 verschieden (% der Matrizen)", rangemode="tozero")
    return _base(fig, 320)


def build_risk_curve(rows):
    """Anteil Platz 1 verschieden über ρ/Spanne (logarithmisch, kleinere Werte = stärker gekrümmt)."""
    xs = [r.rho_norm for r in rows]
    ys = [100.0 * r.changed / r.total for r in rows]
    err = [SW.standard_error_points(r.changed, r.total) for r in rows]
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="lines+markers", error_y=dict(type="data", array=err), line=dict(color=C.COLOR_MULT, width=3),
                               hovertemplate="ρ/Spanne = %{x}: %{y:.1f} %<extra></extra>"))
    fig.add_vline(x=C.RHO_STAR_SINGLE / C.GAIN_SPAN, line_dash="dot", line_color="rgba(120,120,120,0.8)", annotation_text="Kapazitätsfall 1,85", annotation_position="top")
    fig.update_xaxes(title_text="ρ geteilt durch die Spanne (links: stark gekrümmt)", type="log", autorange="reversed")
    fig.update_yaxes(title_text="Platz 1 ändert sich (% der Probleme)", rangemode="tozero")
    return _base(fig, 320)


def build_coupling_bars(rows):
    """Anteil Platz 1 verschieden je Σk für drei Paarvergleiche der Kopplungen (multiplikativ) und additiv (gleichläufig gegen gegenläufig)."""
    xs = [f"Σk = {r.sum_k:g}".replace(".", ",") for r in rows]
    fig = go.Figure()
    series = [("gleichläufig ↔ gegenläufig", [r.same_vs_opposite[0] for r in rows], C.COLOR_MULT), ("gleichläufig ↔ unabhängig", [r.same_vs_independent[0] for r in rows], "#e0a030"),
              ("gegenläufig ↔ unabhängig", [r.opposite_vs_independent[0] for r in rows], "#7b4aa0"), ("additiv: gleichläufig ↔ gegenläufig", [r.add_same_vs_opposite[0] for r in rows], C.COLOR_ADD)]
    for name, counts, color in series:
        fig.add_trace(go.Bar(x=xs, y=[100.0 * c / r.total for c, r in zip(counts, rows)], name=name, marker_color=color, hovertemplate="%{x}: %{y:.1f} %<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Platz 1 ändert sich (% der Probleme)", rangemode="tozero")
    return _base(fig, 340)
