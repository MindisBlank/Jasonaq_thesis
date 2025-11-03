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

from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional, Union
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
    Normalize the 'timebase' arg to a prioritized list of strings to try.
    Accepts:
      - single int seconds (e.g., 900)
      - single label/short (e.g., "15m", "10 minutes")
      - iterable of any mix above (e.g., ["15m","10m","1m"])
    Returns list of canonical short labels like ["15m","10m","1m"] in priority order.
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

    # If only one was provided, append sensible fallbacks
    if len(labels) == 1:
        first = labels[0]
        if first == "15m":
            labels += ["10m", "1m"]
        elif first == "10m":
            labels += ["15m", "1m"]
        elif first == "1m":
            labels += ["15m", "10m"]
        else:
            # Unknown cadence: still add common fallbacks
            labels += ["15m", "10m", "1m"]

    # Deduplicate but keep order
    seen, out = set(), []
    for x in labels:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


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
    Build the URL and GET the JSON. Supports **fallback timebases**:
      - Pass a single preferred cadence (e.g., "15m"), and it will auto-try "10m", then "1m".
      - Or pass an explicit priority list, e.g., ["15m","10m","1m"].

    Returns parsed JSON dict. If dry_run=True, returns {"__urls__": [...]} with the tried URLs.
    Raises GridVisClientError if all attempts fail or return non-JSON.
    """
    cadences = _expand_timebases(timebase)
    tried_urls = []
    last_error = None
    saw_nonjson = False  # <── new flag

    for tb in cadences:
        url = build_hist_url(
            device_id=device_id,
            variable_backend=variable_backend,
            phase_backend=phase_backend,
            timebase=tb,              # <- minutes/labels are converted to seconds inside build_hist_url
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
                return resp.json()  # ✅ success
            except ValueError:
                # Not JSON, probably means "no data"
                saw_nonjson = True
                continue
        except requests.exceptions.RequestException as e:
            last_error = GridVisClientError(f"HTTP error for timebase '{tb}': {e}\nURL: {url}")
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
    DEVICE_ID        = 301
    VARIABLE_BACKEND = "I_Effective"
    PHASE_BACKEND    = "Input04"
    TIMEBASE         = "15m"
    START            = "2025-10-01 12:00"
    END              = "2025-10-02 12:00"
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
