#!/usr/bin/env python3
# janitza_value_fetch.py
"""
CAPABILITY PHASE: Build a measurement catalog for all Janitza devices.

Reads:   metadata/devices.csv  (output of janitza_device_fetch.py)
Calls:   GET /rest/1/projects/{PROJECT}/devices/{id}/hist/values
Writes:  metadata/capabilities.csv
         metadata/capabilities.parquet     (optional)
         metadata/raw_capabilities/{id}.json or .xml (optional raw)

Handles BOTH JSON and XML responses from /hist/values.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
import pathlib
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import xml.etree.ElementTree as ET

# ───────────────────────── STATIC CONFIG — EDIT ME ─────────────────────────
BASE_URL    = "http://gridvis-01.or.is:8070"   # no trailing slash
PROJECT     = "Veitur"
TIMEOUT_S   = 30

INVENTORY_CSV        = "metadata/devices.csv"
OUT_DIR              = "metadata"
OUT_CAP_CSV          = "capabilities.csv"
OUT_CAP_PARQUET      = "capabilities.parquet"      # set to None to skip Parquet
SAVE_RAW             = True
RAW_CAP_DIR          = "metadata/raw_capabilities"
REQUESTS_PER_SECOND  = 3.0
MAX_RETRIES          = 3
RETRY_BACKOFF_SEC    = 2.0
# ───────────────────────────────────────────────────────────────────────────

def rate_limit_sleep(last_call_ts: float, rps: float) -> float:
    min_interval = 1.0 / rps if rps > 0 else 0
    now = time.time()
    wait = min_interval - (now - last_call_ts)
    if wait > 0:
        time.sleep(wait)
        now = time.time()
    return now

def ensure_dir(path: str | pathlib.Path) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def fetch_values(session: requests.Session, device_id: int) -> tuple[str, str]:
    """Return (text, content_type) for /hist/values of a device."""
    url = f"{BASE_URL}/rest/1/projects/{PROJECT}/devices/{device_id}/hist/values"
    resp = session.get(url, timeout=TIMEOUT_S)
    resp.raise_for_status()
    ct = resp.headers.get("Content-Type", "").lower()
    return resp.text, ct

# ========== XML helpers (namespace-agnostic) ==========
def _tag_name(el):
    if el is None:
        return None
    return el.tag.split('}', 1)[-1] if '}' in el.tag else el.tag

def _text(el):
    return el.text.strip() if (el is not None and el.text) else None

def _findall_local(root, local_name):
    return [e for e in root.iter() if _tag_name(e) == local_name]

def _find_first_child_local(parent, local_name):
    if parent is None:
        return None
    for child in list(parent):
        if _tag_name(child) == local_name:
            return child
    return None
# =====================================================

def parse_values_json(txt: str, device_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    JSON format example (as seen in your preview):
    {
      "value": [
        {"id":15769,"timebase":600,"online":false,
         "valueType":{"value":"U_Effective","typeName":"L1","type":"L1","unit":"V","valueName":"Voltage effective"}},
        ...
      ]
    }
    Sometimes it may use "values": [...]
    """
    rows: List[Dict[str, Any]] = []
    try:
        obj = json.loads(txt)
    except json.JSONDecodeError:
        return rows

    arr = None
    if isinstance(obj, dict):
        if isinstance(obj.get("value"), list):
            arr = obj["value"]
        elif isinstance(obj.get("values"), list):
            arr = obj["values"]

    if not arr:
        return rows

    for item in arr:
        if not isinstance(item, dict):
            continue
        vt = item.get("valueType") or {}
        rows.append({
            "device_id": device_row.get("device_id"),
            "dnr_str": device_row.get("dnr_str"),
            "feeder": device_row.get("feeder"),
            "phase_device": device_row.get("phase"),
            "device_type": device_row.get("type"),
            "device_type_name": device_row.get("typeDisplayName"),

            "measurement_id": str(item.get("id")) if item.get("id") is not None else None,
            "online": bool(item.get("online")) if item.get("online") is not None else None,
            "timebase": int(item.get("timebase")) if item.get("timebase") is not None else None,

            "value_backend": vt.get("value"),
            "value_name": vt.get("valueName"),
            "type_backend": vt.get("type"),
            "type_name": vt.get("typeName"),
            "unit": vt.get("unit"),
        })
    return rows

def parse_values_xml(txt: str, device_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Namespace-agnostic XML parser (fallback)."""
    rows: List[Dict[str, Any]] = []
    sniff = txt.strip().lower()
    if sniff.startswith("<html"):
        return rows

    try:
        root = ET.fromstring(txt)
    except ET.ParseError:
        return rows

    value_elems = _findall_local(root, "value")
    if not value_elems:
        return rows

    for val in value_elems:
        val_id_text   = _text(_find_first_child_local(val, "id"))
        online_text   = _text(_find_first_child_local(val, "online"))
        timebase_text = _text(_find_first_child_local(val, "timebase"))

        vt = _find_first_child_local(val, "valueType")
        vt_type       = _text(_find_first_child_local(vt, "type"))       if vt else None
        vt_type_name  = _text(_find_first_child_local(vt, "typeName"))   if vt else None
        vt_unit       = _text(_find_first_child_local(vt, "unit"))       if vt else None
        vt_value      = _text(_find_first_child_local(vt, "value"))      if vt else None
        vt_value_name = _text(_find_first_child_local(vt, "valueName"))  if vt else None

        try:
            timebase = int(timebase_text) if timebase_text else None
        except ValueError:
            timebase = None

        online = None
        if online_text is not None:
            online = online_text.lower() == "true"

        rows.append({
            "device_id": device_row.get("device_id"),
            "dnr_str": device_row.get("dnr_str"),
            "feeder": device_row.get("feeder"),
            "phase_device": device_row.get("phase"),
            "device_type": device_row.get("type"),
            "device_type_name": device_row.get("typeDisplayName"),

            "measurement_id": val_id_text,
            "online": online,
            "timebase": timebase,

            "value_backend": vt_value,
            "value_name": vt_value_name,
            "type_backend": vt_type,
            "type_name": vt_type_name,
            "unit": vt_unit,
        })
    return rows

def parse_values_table_any(txt: str, content_type: str, device_row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Dispatch to JSON or XML parser."""
    ct = (content_type or "").lower()
    if "json" in ct:
        rows = parse_values_json(txt, device_row)
        if rows:
            return rows
        # Fallback: sometimes content-type is json but body invalid—try XML anyway
        return parse_values_xml(txt, device_row)
    else:
        # Some servers mislabel JSON as text/plain → try JSON first
        rows = parse_values_json(txt, device_row)
        if rows:
            return rows
        return parse_values_xml(txt, device_row)

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["__online_rank"] = df["online"].fillna(False).astype(int)
    df["__unit_rank"] = df["unit"].notna().astype(int)
    df.sort_values(
        by=["device_id", "value_backend", "type_backend", "timebase",
            "__online_rank", "__unit_rank"],
        ascending=[True, True, True, True, False, False],
        inplace=True
    )
    deduped = df.drop_duplicates(
        subset=["device_id", "value_backend", "type_backend", "timebase"],
        keep="first"
    )
    return deduped.drop(columns=["__online_rank", "__unit_rank"])

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build capability catalog from /hist/values (JSON or XML).")
    p.add_argument("--inventory", default=INVENTORY_CSV,
                   help="Path to devices.csv from janitza_device_fetch.py")
    p.add_argument("--out-dir", default=OUT_DIR, help="Output directory")
    p.add_argument("--no-save-raw", dest="save_raw", action="store_false")
    p.add_argument("--filter-device-ids", default=None,
                   help="Comma-separated device IDs to include (e.g., '1,2,3').")
    p.add_argument("--limit", type=int, default=None,
                   help="Only process the first N devices after filtering (for testing).")
    p.add_argument("--rps", type=float, default=REQUESTS_PER_SECOND,
                   help="Max requests per second.")
    p.add_argument("--retries", type=int, default=MAX_RETRIES,
                   help="Max HTTP retries per device.")
    return p

def main():
    args = build_arg_parser().parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    out_csv_path = os.path.join(out_dir, OUT_CAP_CSV)
    out_parquet_path = os.path.join(out_dir, OUT_CAP_PARQUET) if OUT_CAP_PARQUET else None


    try:
        inv = pd.read_csv(INVENTORY_CSV, dtype={"dnr_str": "string"})
    except Exception as e:
        print(f"❌ Failed to read inventory CSV at {INVENTORY_CSV}: {e}", file=sys.stderr)
        sys.exit(1)

    df = inv.copy()
    if args.filter_device_ids:
        keep_ids = {int(x.strip()) for x in args.filter_device_ids.split(",") if x.strip()}
        df = df[df["device_id"].isin(keep_ids)]

    if args.limit:
        df = df.head(args.limit)

    if df.empty:
        print("No devices to process after filtering.", file=sys.stderr)
        sys.exit(0)

    session = requests.Session()
    all_rows: List[Dict[str, Any]] = []

    last_call = 0.0

    for _, row in df.iterrows():
        device_id = int(row["device_id"])

        # Rate limit
        last_call = rate_limit_sleep(last_call, args.rps)

        # Fetch with retries
        txt: Optional[str] = None
        content_type: str = ""


        if txt is None:
            for attempt in range(1, args.retries + 1):
                try:
                    txt, content_type = fetch_values(session, device_id)
                    break
                except requests.RequestException as e:
                    if attempt >= args.retries:
                        print(f"❌ Device {device_id}: HTTP error after {attempt} attempts: {e}", file=sys.stderr)
                    else:
                        sleep_s = (RETRY_BACKOFF_SEC ** (attempt - 1))
                        print(f"⚠️ Device {device_id}: HTTP error, retrying in {sleep_s:.1f}s... ({attempt}/{args.retries})", file=sys.stderr)
                        time.sleep(sleep_s)
            else:
                continue
        # Parse capabilities
        rows = parse_values_table_any(txt, content_type, device_row=row.to_dict())
        if not rows:
            preview = (txt[:200] if txt else "").replace("\n", " ")
            print(f"⚠️ Device {device_id}: No measurement entries parsed. Preview: {preview}", file=sys.stderr)
        else:
            all_rows.extend(rows)

    if not all_rows:
        print("No capability rows parsed for any device.", file=sys.stderr)
        sys.exit(2)

    caps = pd.DataFrame(all_rows)
    caps = deduplicate(caps)


    # --- Capability filter (remove noisy variables) ---
    PATTERN = r"(Harmonic Voltage|Interharmonic Voltage|THD|flicker|Harmonic Current|temperature|Distortion power)"
    before_n = len(caps)
    caps = caps[~caps["value_name"].astype(str).str.contains(PATTERN, case=False, na=False)].copy()
    removed_n = before_n - len(caps)
    print(f"🔎 Capability filter: removed {removed_n}, kept {len(caps)}")


    ordered_cols = [
        "device_id", "dnr_str", "feeder", "phase_device",
        "device_type", "device_type_name",
        "value_backend", "value_name", "type_backend", "type_name",
        "unit", "timebase", "online", "measurement_id"
    ]
    ordered_cols += [c for c in caps.columns if c not in ordered_cols]
    caps = caps[ordered_cols]

    try:
        ensure_dir(out_dir)
        caps.to_csv(out_csv_path, index=False, encoding="utf-8")
        print(f"✅ Saved capabilities CSV: {out_csv_path} (rows: {len(caps)})")
    except Exception as e:
        print(f"❌ Could not save capabilities CSV: {e}", file=sys.stderr)

    if out_parquet_path:
        try:
            caps.to_parquet(out_parquet_path, index=False)
            print(f"✅ Saved capabilities Parquet: {out_parquet_path}")
        except Exception as e:
            print(f"⚠️ Skipping Parquet save: {e}", file=sys.stderr)

    summary = {
        "rows": int(len(caps)),
        "devices_covered": int(caps["device_id"].nunique()),
        "unique_variables": sorted(caps["value_backend"].dropna().unique().tolist()),
        "unique_timebases": sorted([int(x) for x in caps["timebase"].dropna().unique().tolist()]),
    }
    print("—" * 60)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
