# plot_janitza.py
from datetime import datetime, timezone
import matplotlib.pyplot as plt

"""
Simple plotting functions for Janitza GridVis historical data JSON.

"""


def _ns_to_dt_utc(ns: int):
    """Convert Unix time in *nanoseconds* to timezone-aware UTC datetime."""
    return datetime.fromtimestamp(ns / 1_000_000_000, tz=timezone.utc)

def extract_series(data: dict, which: str = "avg"):
    """
    Turn GridVis JSON into (timestamps, values).
    `which` ∈ {"avg", "min", "max"}.
    """
    vals = data.get("values", []) or []
    xs, ys = [], []
    for v in vals:
        t_ns = v.get("startTime")  # or endTime; startTime aligns with the bucket start
        y = v.get(which)
        if t_ns is None or y is None:
            continue
        xs.append(_ns_to_dt_utc(t_ns))
        ys.append(float(y))
    return xs, ys

def simple_line_plot(data: dict, which: str = "avg", title: str | None = None):
    """
    Plot a single line (avg/min/max) from the Janitza JSON.
    Returns the Matplotlib figure.
    """
    xs, ys = extract_series(data, which=which)
    if not xs:
        print("No points to plot.")
        return None

    vt = data.get("valueType", {}) or {}
    unit = vt.get("unit", "")
    name = vt.get("valueName") or vt.get("value") or "Measurement"

    fig = plt.figure()
    plt.plot(xs, ys, marker=".", linewidth=1)
    plt.xlabel("Time (UTC)")
    ylabel = f"{name} [{unit}]" if unit else name
    plt.ylabel(ylabel)
    plt.title(title or f"{name} – {which}")
    plt.tight_layout()
    return fig

def line_with_band_plot(data: dict, title: str | None = None):
    """
    Optional: avg line with min/max band.
    """
    xs_avg, ys_avg = extract_series(data, which="avg")
    xs_min, ys_min = extract_series(data, which="min")
    xs_max, ys_max = extract_series(data, which="max")

    if not xs_avg:
        print("No points to plot.")
        return None

    vt = data.get("valueType", {}) or {}
    unit = vt.get("unit", "")
    name = vt.get("valueName") or vt.get("value") or "Measurement"

    fig = plt.figure()
    plt.plot(xs_avg, ys_avg, marker=".", linewidth=1, label="avg")
    if xs_min and xs_max and len(xs_min) == len(xs_max) == len(xs_avg):
        # Don’t set colors; let matplotlib pick defaults
        plt.fill_between(xs_avg, ys_min, ys_max, alpha=0.2, label="min–max")

    plt.xlabel("Time (UTC)")
    ylabel = f"{name} [{unit}]" if unit else name
    plt.ylabel(ylabel)
    plt.title(title or f"{name} – avg with min/max band")
    plt.legend()
    plt.tight_layout()
    return fig
