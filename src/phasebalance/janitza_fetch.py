#!/usr/bin/env python3
# gridvis_hist_client.py
"""
GridVis (Janitza) helper for building the historical-data URL and fetching JSON.

Public API:
- build_hist_url(...)
- fetch_hist_json(...)

All functions rely on the static config below.

Must be connected to Or.is VPN or run on-site to access GridVis server.
"""
#!!!NOTICE : When pulling a device at sample rate of 60s the timestamp data does not align with the minute for example if you change
#!!! the UTC ns time to datetime it would look like 11:59:00.00004 instead of 12:00:00.00000. 
#!!! I am not sure how I will handle this sohuld I shift it here or later in the data processing pipeline..
from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

# ─────────────────────── Static config (do not change elsewhere) ───────────────────────
BASE_URL       = "http://gridvis-01.or.is:8070"   # no trailing slash
PROJECT        = "Veitur"
TIMEZONE       = "Atlantic/Reykjavik"             # how human datetimes are interpreted
HTTP_TIMEOUT_S = 30
# ───────────────────────────────────────────────────────────────────────────────────────

# Optional: one session for connection reuse when calling many times
_SESSION = requests.Session()

_TIMEBASE_LABELS = {
    60: "1 minute",
    600: "10 minutes",
    900: "15 minutes",
    3600: "1 hour",
}
_LABEL_TO_TIMEBASE = {v.lower(): k for k, v in _TIMEBASE_LABELS.items()}


class GridVisClientError(Exception):
    """Raised for input/HTTP/parsing errors in gridvis_hist_client."""

def _is_energy_unit(unit: str | None) -> bool:
    if not unit:
        return False
    u = unit.strip().lower()
    return u in {"wh", "varh", "kwh", "kvarh", "mwh", "mvarh"}

def _parse_timebase(tb: Union[int, str]) -> int:
    """
    Accepts:
      - int seconds (60/600/900/3600)
      - string seconds ("900")
      - labels ("15 minutes", "1 hour")
      - shorthands ("15m", "1h")
    Returns seconds as int.
    """
    if isinstance(tb, int):
        return tb
    s = str(tb).strip().lower()
    if s.isdigit():
        return int(s)
    if s in _LABEL_TO_TIMEBASE:
        return _LABEL_TO_TIMEBASE[s]
    m = re.fullmatch(r"(\d+)\s*([mh])", s)
    if m:
        val = int(m.group(1))
        return val * 60 if m.group(2) == "m" else val * 3600
    raise GridVisClientError(f"Unrecognized timebase: {tb!r}")

def _to_epoch_ms(
    dt_in: Union[str, datetime],
    input_tz: str = TIMEZONE
) -> int:
    """
    Convert a datetime-like input to UTC epoch ms.
    Accepts:
      - datetime (naive assumed in input_tz; aware kept as-is then converted to UTC)
      - strings:
          * ISO-ish: "YYYY-MM-DD", "YYYY-MM-DD HH:MM", "...:SS", "YYYY-MM-DDTHH:MM[:SS]"
          * relative: "-24h", "-7d", "-2w"
          * keyword:  "now"
    NOTE: For strings like '2025-10-02', use zero-padded month/day.
    """
    if isinstance(dt_in, datetime):
        dt_local = dt_in if dt_in.tzinfo else dt_in.replace(tzinfo=ZoneInfo(input_tz))
    else:
        s = dt_in.strip().lower()
        if s == "now":
            dt_local = datetime.now(ZoneInfo(input_tz))
        else:
            rel = re.fullmatch(r"-\s*(\d+)\s*([hdw])", s)
            if rel:
                qty = int(rel.group(1))
                unit = rel.group(2)
                now_local = datetime.now(ZoneInfo(input_tz))
                delta = {"h": timedelta(hours=qty), "d": timedelta(days=qty), "w": timedelta(weeks=qty)}[unit]
                dt_local = now_local - delta
            else:
                # Try common ISO-ish formats (zero-padded)
                for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
                            "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S",
                            "%Y-%m-%d"):
                    try:
                        dt_local = datetime.strptime(s, fmt).replace(tzinfo=ZoneInfo(input_tz))
                        break
                    except ValueError:
                        continue
                else:
                    raise GridVisClientError(f"Unrecognized datetime: {dt_in!r} "
                                             f"(use zero-padded form like '2025-10-02 12:00').")
    dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
    return int(dt_utc.timestamp() * 1000)

def build_hist_url(
    *,
    device_id: int,
    variable_backend: str,
    phase_backend: str,
    timebase: Union[int, str],
    start: Union[str, datetime],
    end: Union[str, datetime],
    tz_name: str = TIMEZONE,
) -> str:
    """
    Build the exact GridVis historical values URL your dashboard uses.
    """
    tb_sec   = _parse_timebase(timebase)
    start_ms = _to_epoch_ms(start, tz_name)
    end_ms   = _to_epoch_ms(end,   tz_name)
    return (
        f"{BASE_URL}/rest/1/projects/{PROJECT}/devices/{device_id}/hist/values/"
        f"{variable_backend}/{phase_backend}/{tb_sec}.json"
        f"?start=UTC_{start_ms}&end=UTC_{end_ms}&timezone={tz_name}"
    )

def _expand_timebases(tb: Union[int, str, list, tuple]) -> list[str]:
    """
    Normalize the 'timebase' arg to a prioritized list of short labels to try.

    - If a single cadence is given, that cadence is tried first.
    - If a list is given, its order is respected.
    - '1m' is always appended as a final fallback (unless already present).

    Returns list of canonical short labels like ["15m", "1m"] in priority order.
    """
    def _to_short_label(x) -> str:
        sec = _parse_timebase(x)                 # uses your existing helper
        if sec % 3600 == 0: return f"{sec//3600}h"
        if sec % 60   == 0: return f"{sec//60}m"
        return f"{sec}s"

    # If a single value, make it a list
    if not isinstance(tb, (list, tuple)):
        tbs = [tb]
    else:
        tbs = list(tb)

    labels = [_to_short_label(x) for x in tbs]

    # Always ensure 1m is a final fallback, unless the caller already included it
    if "1m" not in labels:
        labels.append("1m")

    # Deduplicate but keep order
    seen = set()
    out: list[str] = []
    for x in labels:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _aggregate_from_1min(data: Dict[str, Any], target_tb_sec: int) -> Dict[str, Any]:
    """
    Aggregate 1-minute series up to a coarser fixed timebase (e.g. 15 min).

    Assumes:
      - input JSON has the same structure as GridVis hist values:
          {
            "timebase": 60,
            "values": [
              {
                "max": ...,
                "min": ...,
                "avg": ...,
                "startTime": ...,
                "endTime": ...
              }, ...
            ]
          }
      - 'avg' is an average-like quantity over each interval (e.g. A, V, kW),
        so the coarser 'avg' is the simple mean of the per-minute 'avg' values.
    """
    base_tb_sec = int(data.get("timebase", 60))
    if base_tb_sec != 60:
        raise GridVisClientError(
            f"_aggregate_from_1min expected 60 s base timebase, got {base_tb_sec}"
        )

    if target_tb_sec == base_tb_sec:
        return data

    if target_tb_sec < base_tb_sec or target_tb_sec % base_tb_sec != 0:
        raise GridVisClientError(
            f"Cannot aggregate from 1m to {target_tb_sec} s timebase"
        )

    step = target_tb_sec // base_tb_sec
    values = list(data.get("values") or [])

    if not values:
        # Nothing to aggregate; just adjust the reported timebase
        new_data = dict(data)
        new_data["timebase"] = target_tb_sec
        new_data["values"] = []
        new_data["__resampled__"] = True
        return new_data

    # Ensure values are in chronological order
    values.sort(key=lambda v: v.get("startTime", 0))

    unit = (data.get("valueType") or {}).get("unit")
    energy_mode = _is_energy_unit(unit)

    aggregated: list[Dict[str, Any]] = []

    for i in range(0, len(values), step):
        chunk = values[i:i + step]
        if len(chunk) < step:
            # Drop incomplete trailing window
            break
        
        first = chunk[0]
        last = chunk[-1]

        agg_entry: Dict[str, Any] = {}

        # Preserve time bounds
        if "startTime" in first:
            agg_entry["startTime"] = first["startTime"]
        if "endTime" in last:
            agg_entry["endTime"] = last["endTime"]

        if energy_mode:
            # ENERGY REGISTERS: avg behaves like a cumulative counter → keep last value
            last_avg = None
            for v in reversed(chunk):
                if v.get("avg") is not None:
                    last_avg = v.get("avg")
                    break
            if last_avg is not None:
                agg_entry["avg"] = float(last_avg)

            # Optional: keep min/max absent for energy (often NaN anyway)
            aggregated.append(agg_entry)
            continue

        # Aggregate max / min / avg if present
        max_vals = [v["max"] for v in chunk if "max" in v and v["max"] is not None]
        if max_vals:
            agg_entry["max"] = max(max_vals)

        min_vals = [v["min"] for v in chunk if "min" in v and v["min"] is not None]
        if min_vals:
            agg_entry["min"] = min(min_vals)

        avg_vals = [v["avg"] for v in chunk if "avg" in v and v["avg"] is not None]
        if avg_vals:
            agg_entry["avg"] = sum(avg_vals) / len(avg_vals)

        # If you later need other fields (e.g. "sum"), add similar logic here.

        aggregated.append(agg_entry)

    new_data = dict(data)
    new_data["timebase"] = target_tb_sec
    new_data["values"] = aggregated
    new_data["__resampled__"] = True
    return new_data

def fetch_hist_json(
    *,
    device_id: int,
    variable_backend: str,
    phase_backend: str,
    timebase: Union[int, str, list, tuple],
    start: Union[str, datetime],
    end: Union[str, datetime],
    tz_name: str = TIMEZONE,
    auth_token: Optional[str] = None,
    dry_run: bool = False,
    timeout_s: int = HTTP_TIMEOUT_S,
) -> Optional[Dict[str, Any]]:
    """
    Build the URL and GET the JSON. Supports **fallback timebases** with automatic
    1-minute resampling:

      - Pass a single preferred cadence (e.g., "15m").
        The client will first try native "15m" data.
        If unavailable, it will try "1m" and, if found, aggregate 1m → 15m.
      - Or pass an explicit priority list, e.g., ["15m","10m"] — "1m" will
        still be added as a final fallback and, if used, will be aggregated
        up to the first requested cadence.

    Returns parsed JSON dict. If dry_run=True, returns {"__urls__": [...]} with
    the tried URLs.

    Guarantees:
      - If it returns non-None and you requested a single cadence, the returned
        JSON's "timebase" will match what you asked for (e.g., 900 s for "15m"),
        even if the underlying server data came from 1-minute samples.
    """
    # Determine the *primary* requested timebase for resampling decisions
    if isinstance(timebase, (list, tuple)) and timebase:
        primary_tb = timebase[0]
    else:
        primary_tb = timebase

    target_tb_sec = _parse_timebase(primary_tb)

    cadences = _expand_timebases(timebase)
    tried_urls: list[str] = []
    last_error: Optional[GridVisClientError] = None
    saw_nonjson = False

    for tb in cadences:
        url = build_hist_url(
            device_id=device_id,
            variable_backend=variable_backend,
            phase_backend=phase_backend,
            timebase=tb,  # labels ("15m","1m") are parsed inside build_hist_url
            start=start,
            end=end,
            tz_name=tz_name,
        )
        tried_urls.append(url)

        if dry_run:
            # For inspection, show every URL we would try
            continue

        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        try:
            resp = _SESSION.get(url, headers=headers, timeout=timeout_s)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                # Not JSON, probably means "no data"
                saw_nonjson = True
                continue

            attempt_tb_sec = _parse_timebase(tb)

            # 1) Exact match to the primary requested cadence → return as-is
            if attempt_tb_sec == target_tb_sec:
                data["__resampled__"] = False
                return data

            # 2) Fallback from 1-minute to a coarser cadence (e.g., 1m → 15m)
            if attempt_tb_sec == 60 and target_tb_sec >= 60 and target_tb_sec % 60 == 0:
                return _aggregate_from_1min(data, target_tb_sec)

            # 3) Any other successful cadence is returned as-is.
            #    This only happens if the caller explicitly asked for that cadence
            #    in the 'timebase' list.
            data["__resampled__"] = False
            return data

        except requests.exceptions.RequestException as e:
            last_error = GridVisClientError(
                f"HTTP error for timebase '{tb}': {e}\nURL: {url}"
            )
            continue

    # ─────────── post-loop logic ───────────
    if dry_run:
        return {"__urls__": tried_urls}

    if saw_nonjson:
        print(f"⚠️ No data for device {device_id} ({phase_backend}), skipping.")
        return None

    if last_error:
        # Serious network or HTTP issue, not just "no data"
        raise last_error

    # Fallback catch-all
    print(f"⚠️ No data for device {device_id} ({phase_backend}), skipping.")
    return None




# Optional: quick smoke test when running this file directly
if __name__ == "__main__":
    # Edit these lines to test quickly inside VS Code.
    DEVICE_ID        = 365 #262 , 263,
    VARIABLE_BACKEND = "ActiveEnergy"
    PHASE_BACKEND    = "SUM13"
    TIMEBASE         = "1 hour"
    START            = "2025-11-01 12:00"
    END              = "2025-11-01 14:00"
    AUTH_TOKEN       = None

    try:
        out = fetch_hist_json(
            device_id=DEVICE_ID,
            variable_backend=VARIABLE_BACKEND,
            phase_backend=PHASE_BACKEND,
            timebase=TIMEBASE,
            start=START,
            end=END,
            auth_token=AUTH_TOKEN,
            dry_run=False,   # set True to print only the URL
        )
        # Pretty print
        print(json.dumps(out, indent=2, ensure_ascii=False))
    except GridVisClientError as e:
        print(f"ERROR: {e}")
