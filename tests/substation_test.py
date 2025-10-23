#!/usr/bin/env python3
# janitza_fetch_simple.py
import json
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import requests

# ───────────────────────── CONFIG — EDIT ME ─────────────────────────
BASE_URL         = "http://gridvis-01.or.is:8070"   # no trailing slash
PROJECT          = "Veitur"
DEVICE_ID        = 301                              # <- Janitza device id (int)
VARIABLE_BACKEND = "I_Effective"            # <- backend "value"
PHASE_BACKEND    = "Input04"                              # <- backend "type" (e.g., L1/L2/L3 or API type)
TIMEBASE         = "15m"                             # seconds (900), label ("15 minutes"), or shorthand ("15m","1h")
START            = "2025-10-01 12:00"                # "YYYY-MM-DD HH:MM", "YYYY-MM-DD", or relative like "-24h"
END              = "2025-10-2 12:00"                # same formats as START
TIMEZONE         = "Atlantic/Reykjavik"              # how START/END are interpreted
AUTH_TOKEN       = None                              # e.g., "eyJhbGciOi..." or None
DRY_RUN          = False                             # True = only print URL, don’t call server
HTTP_TIMEOUT_S   = 30
# ────────────────────────────────────────────────────────────────────

TIMEBASE_LABELS = {
    60: "1 minute",
    600: "10 minutes",
    900: "15 minutes",
    3600: "1 hour",
}
LABEL_TO_TIMEBASE = {v.lower(): k for k, v in TIMEBASE_LABELS.items()}

def parse_timebase(tb_str_or_int) -> int:
    """
    Accepts:
      - int seconds (60/600/900/3600)
      - string seconds ("900")
      - labels ("15 minutes", "1 hour")
      - shorthands ("15m", "1h")
    """
    if isinstance(tb_str_or_int, int):
        return tb_str_or_int
    s = str(tb_str_or_int).strip().lower()
    if s.isdigit():
        return int(s)
    if s in LABEL_TO_TIMEBASE:
        return LABEL_TO_TIMEBASE[s]
    m = re.fullmatch(r"(\d+)\s*([mh])", s)
    if m:
        val = int(m.group(1))
        return val * 60 if m.group(2) == "m" else val * 3600
    raise ValueError(f"Unrecognized timebase: {tb_str_or_int}")

def to_epoch_ms(dt_str: str, input_tz: str) -> int:
    """
    Parse a datetime string in input_tz and convert to UTC epoch ms.
    Supports relative '-24h', '-7d', '-2w'.
    """
    s = dt_str.strip()
    rel = re.fullmatch(r"-\s*(\d+)\s*([hdw])", s, flags=re.I)
    if rel:
        qty = int(rel.group(1))
        unit = rel.group(2).lower()
        now_local = datetime.now(ZoneInfo(input_tz))
        delta = {"h": timedelta(hours=qty), "d": timedelta(days=qty), "w": timedelta(weeks=qty)}[unit]
        dt_local = now_local - delta
    else:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M",
                    "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d"):
            try:
                dt_local = datetime.strptime(s, fmt).replace(tzinfo=ZoneInfo(input_tz))
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unrecognized datetime format: {dt_str}")
    dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
    return int(dt_utc.timestamp() * 1000)

def build_url(
    base_url: str,
    project: str,
    device_id: int,
    variable_backend: str,
    phase_backend: str,
    timebase_seconds: int,
    start_epoch_ms: int,
    end_epoch_ms: int,
    tz_name: str,
) -> str:
    return (
        f"{base_url}/rest/1/projects/{project}/devices/{device_id}/hist/values/"
        f"{variable_backend}/{phase_backend}/{timebase_seconds}.json"
        f"?start=UTC_{start_epoch_ms}&end=UTC_{end_epoch_ms}&timezone={tz_name}"
    )

def main():
    tb = parse_timebase(TIMEBASE)
    start_ms = to_epoch_ms(START, TIMEZONE)
    end_ms = to_epoch_ms(END, TIMEZONE)

    url = build_url(
        BASE_URL, PROJECT, DEVICE_ID,
        VARIABLE_BACKEND, PHASE_BACKEND, tb,
        start_ms, end_ms, TIMEZONE
    )

    print("Request URL:")
    print(url)

    if DRY_RUN:
        return

    headers = {}
    if AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {AUTH_TOKEN}"

    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT_S)
    try:
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\nHTTP error: {e}")
        print("Response text:\n", resp.text)
        return

    try:
        data = resp.json()
        print("\nJSON response:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except ValueError:
        print("\nNon-JSON response:")
        print(resp.text)

if __name__ == "__main__":
    main()
