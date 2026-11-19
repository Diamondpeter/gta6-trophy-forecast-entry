"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 GTA VI trophy, five quantiles of its Exophase earn
rate at the case's target, from the house's reads of the trophy page.

Method — counts, not just rates.  A rate is earned players over players
tracked, and both move: players already tracked keep earning trophies, while
new players joining Exophase dilute every rate.  So the model fits the earned
counts E_i(t) and the tracked-player count N(t) separately, each as a power
law in log time since release (log E and log N linear in u = ln(1 + hours
since 2026-11-19T00:00Z)), on the distinct reads (consecutive identical
reads are one stale page, not new information), and divides them at the
target.  Slopes are shrunk toward the cross-trophy mean and damped, and
forecasts are capped so no trophy's earned count exceeds the tracked base.

The stale page.  Exophase refreshes a game's counts on its own schedule,
hours apart, so a flat stretch of reads is a stale page, not a still game.
When the history holds only one distinct read, a launch-dynamics prior
stands in, calibrated on the first refresh the house documented (638 to
3,868 players tracked, 943 to 4,276 trophies earned): the tracked base keeps
growing like (1 + t)^1.06, newly tracked players hold each trophy at about
the square of its current rate (a trophy at 1% is held by ~0.01% of
newcomers), and the players already tracked keep earning at a modest rate.
Rates therefore fall at the next refresh — persistence would miss that.

Spread.  The 5-95% band comes from the fit's uncertainty, the extrapolation
distance in u and the refresh timing, so about 90% of rates should land
inside it; the pinball loss pays for true quantiles.  Every row is clamped
to 0.01-99.99%, non-decreasing, one row per trophy: the program always
writes a complete, valid forecast.

Python 3.13, standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# --- priors, calibrated on the house's documented first refresh ----------
# log N slope against u = ln(1 + hours since release): 638 -> 3,868 players
# tracked while u went 0 -> ln(5.5), so ln(6.06)/ln(5.5) ~= 1.06.
BETA_PRIOR = 1.06
# Newly tracked players' earn rate for a trophy at rate r (percent):
# q = 100 * (r/100) ** GAMMA_NEW.  GAMMA_NEW = 2 fits the documented
# refresh totals (1,478 -> 1,105 mean trophies per tracked player).
GAMMA_NEW = 2.0
# Old cohort's earned-count growth over a stale window, per trophy:
# G = 1 + GROWTH_OLD * (1 - r/100): rare trophies gain most, common ones
# barely move (everyone already holds them).
GROWTH_OLD = 0.5

# --- spread (logit sd) ----------------------------------------------------
SPREAD_BASE = 0.45      # model error, rounding, regime change
SPREAD_SLOPE = 0.45     # slope error, per unit of extrapolation in u
SPREAD_REFRESH = 0.30   # the answer read's staleness vs the target
SLOPE_DAMP = 0.8         # shrink extrapolated growth toward flat
FIT_WINDOW_HOURS = 18.0   # fit on the recent distinct reads: launch growth
                          # decelerates, so the current regime is what counts


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp(pct)
    return math.log(p / (100 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as the house baseline rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Acklam's rational approximation to the standard normal quantile."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low, p_high = 0.02425, 1 - 0.02425
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tau_hours(when: datetime) -> float:
    """Hours since release (2026-11-19T00:00Z), clamped at 0."""
    return max(0.0, (when - RELEASE).total_seconds() / 3600.0)


def u_of(tau: float) -> float:
    return math.log1p(tau)


def linfit(xs, ys, ws=None):
    """Weighted least squares y ~ a + b*x. Returns (a, b)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if ws is None:
        ws = [1.0] * n
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return my - b * mx, b


def distinct_reads(history):
    """Collapse consecutive identical reads: one stale page, one snapshot."""
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key != prev:
            out.append(read)
            prev = key
    return out


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = [Z[LEVELS.index(t)] if t in LEVELS else norm_ppf(t) for t in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    target = parse_time(case["target"])
    tau_t = tau_hours(target)
    u_t = u_of(tau_t)

    if not history:
        # No reads at all: say 50% with a very wide band, but stay valid.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(50) + z * 2.0)) for z in zs]}
            for t in trophies]}

    last = history[-1]
    n_last = last.get("players_tracked") or 0
    rates_last = last.get("rates") or []
    # Last non-null rate per trophy (the last read usually has them all).
    last_rate = []
    for i in range(len(trophies)):
        r = None
        for read in reversed(history):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                r = rates[i]
                break
        last_rate.append(r if r is not None else 1.0)

    distinct = distinct_reads(history)
    tau_last = tau_hours(parse_time(last["at"]))
    u_anchor = u_of(tau_last)

    if len(distinct) >= 2:
        # ---- data-driven: power-law fits on the recent distinct reads ----
        checkpoint = parse_time(case["at"])
        tau_cp = tau_hours(checkpoint)
        window = [r for r in distinct
                  if tau_hours(parse_time(r["at"])) >= tau_cp - FIT_WINDOW_HOURS]
        if len(window) < 3:
            window = distinct
        us = [u_of(tau_hours(parse_time(r["at"]))) for r in window]
        # Recency weights: the latest read counts double the earliest.
        ws = [1.0 + (u - us[0]) / (us[-1] - us[0] + 1e-9) for u in us]
        u_ref = sum(w * u for w, u in zip(ws, us)) / sum(ws)

        # Tracked players.
        log_n = [math.log(max(1.0, r.get("players_tracked") or 1)) for r in window]
        a_n, b_n_raw = linfit(us, log_n, ws)
        # Anchor on the fitted value at u_ref (raw slope), then extrapolate
        # with the damped slope: damping the slope must not move the anchor.
        log_n_ref = a_n + b_n_raw * u_ref
        b_n = min(3.0, max(0.0, SLOPE_DAMP * b_n_raw))
        n_target = math.exp(log_n_ref + b_n * (u_t - u_ref))
        n_target = max(n_target, n_last)

        # Earned counts per trophy, slopes shrunk toward their mean.
        raw_slopes = []
        e_refs = []
        for i in range(len(trophies)):
            e = []
            for r in window:
                rates = r.get("rates") or []
                rate = rates[i] if i < len(rates) and rates[i] is not None else None
                n = r.get("players_tracked") or 0
                e.append((rate / 100.0 * n) if rate is not None else None)
            pts = [(u, math.log(max(0.5, v))) for u, v in zip(us, e) if v is not None]
            if len(pts) >= 2:
                a_i, b_i = linfit([p[0] for p in pts], [p[1] for p in pts])
            elif len(pts) == 1:
                a_i, b_i = pts[0][1], 0.0
            else:
                a_i, b_i = math.log(0.5), 0.0
            raw_slopes.append(b_i)
            e_refs.append(math.exp(a_i + b_i * u_ref))
        # E-weighted mean slope for shrinkage.
        wsum = sum(e_refs)
        b_bar = sum(b * e for b, e in zip(raw_slopes, e_refs)) / wsum if wsum > 0 else 0.0
        n_pts = len(window)
        shrink = n_pts / (n_pts + 2.0)
        medians = []
        for i in range(len(trophies)):
            b_i = shrink * raw_slopes[i] + (1 - shrink) * b_bar
            b_i = min(3.0, max(0.0, SLOPE_DAMP * b_i))
            e_t = e_refs[i] * math.exp(b_i * (u_t - u_ref))
            e_last_i = last_rate[i] / 100.0 * n_last
            e_t = max(e_t, e_last_i)          # earned counts only grow
            e_t = min(e_t, n_target)          # cannot exceed players tracked
            medians.append(clamp(100.0 * e_t / n_target) if n_target > 0 else last_rate[i])
        du = u_t - u_ref
    else:
        # ---- flat history: the page is stale; use the launch prior -------
        first = distinct[0]
        tau_first = tau_hours(parse_time(first["at"]))
        mult = ((1.0 + tau_t) / (1.0 + tau_first)) ** BETA_PRIOR
        n_target = max(n_last * mult, n_last)
        new_players = max(0.0, n_target - n_last)
        medians = []
        for i in range(len(trophies)):
            r = last_rate[i]
            g = 1.0 + GROWTH_OLD * (1.0 - r / 100.0)
            q_new = 100.0 * (r / 100.0) ** GAMMA_NEW
            e_t = (r / 100.0 * n_last) * g + new_players * q_new / 100.0
            e_t = min(e_t, n_target)
            medians.append(clamp(100.0 * e_t / n_target) if n_target > 0 else r)
        du = u_t - u_of(tau_first)

    spread = math.sqrt(SPREAD_BASE ** 2 + (SPREAD_SLOPE * du) ** 2 + SPREAD_REFRESH ** 2)
    rows = []
    for t, med in zip(trophies, medians):
        lg = logit(med)
        qs = [round4(clamp(inv_logit(lg + z * spread))) for z in zs]
        # Rounding can only tie neighbours, never invert them; enforce order.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
