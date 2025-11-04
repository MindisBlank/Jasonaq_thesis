#!/usr/bin/env python3
# janitza_parse.py
"""
Inventory Phase: Fetch and parse device metadata from Janitza GridVis REST API.

Fetches:
    GET /rest/1/projects/{PROJECT}/devices

Outputs:
    metadata/devices.csv
    metadata/devices.parquet   (optional)
"""

from __future__ import annotations
import os
import re
import sys
import json
import pathlib
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional

# ───────────────────────── STATIC CONFIG — EDIT ME ─────────────────────────
BASE_URL   = "http://gridvis-01.or.is:8070"   # no trailing slash
PROJECT    = "Veitur"
TIMEOUT_S  = 30

# Where to save outputs
OUT_DIR            = "metadata"
CSV_FILENAME       = "devices.csv"
PARQUET_FILENAME   = "devices.parquet"   # set to None to skip Parquet
SAVE_RAW_XML       = True
RAW_XML_FILENAME   = "devices_raw.xml"

# If True, also call /devices/{id}. Usually unnecessary.
FETCH_DETAILS_PER_DEVICE = False
# ──────────────────────────────────────────────────────────────────────────

def ensure_dir(path: str | pathlib.Path) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def fetch_devices_xml(session: requests.Session) -> str:
    url = f"{BASE_URL}/rest/1/projects/{PROJECT}/devices"
    resp = session.get(url, timeout=TIMEOUT_S)
    resp.raise_for_status()
    return resp.text


def parse_location_from_text(text: str | None) -> Dict[str, Any]:
    """
    Extract substation info from description/name like:
    'D0029 Borgartún 20 SP1 L2' → DNR=29, feeder=SP1, phase=L2
    """
    res = {"dnr_str": None, "feeder": None, "phase": None}
    if not text:
        return res

    # DNR like D0029, D034, D29, etc.
    dnr_m = re.search(r"\bD0*(\d{1,4})\b", text, flags=re.IGNORECASE)
    if dnr_m:
        res["dnr_str"] = f"D{int(dnr_m.group(1)):04d}"

    # Feeder like SP1 / SP2 / SPxx
    feeder_m = re.search(r"\bSP(\d{1,2})\b", text, flags=re.IGNORECASE)
    if feeder_m:
        res["feeder"] = f"SP{feeder_m.group(1)}"

    # Phase like L1 / L2 / L3
    phase_m = re.search(r"\bL([123])\b", text, flags=re.IGNORECASE)
    if phase_m:
        res["phase"] = f"L{phase_m.group(1)}"

    return res


def parse_devices_table(xml_text: str) -> List[Dict[str, Any]]:
    """
    Parse the <devices> XML into a list of dict rows.
    Expected snippet:
      <device>
        <description>D0029 Borgartún 20 SP1 L2</description>
        <id>1</id>
        <name>D0029 SP1 L2</name>
        <serialNr>1743-5394</serialNr>
        <type>JanitzaUMG96RME</type>
        <typeDisplayName>UMG 96RM-E</typeDisplayName>
      </device>
    """
    rows: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        raise RuntimeError(f"Failed to parse /devices XML: {e}") from e

    for dev in root.findall(".//device"):
        def _get(tag: str) -> Optional[str]:
            elem = dev.find(tag)
            return elem.text.strip() if elem is not None and elem.text else None

        device_id_str   = _get("id")
        name            = _get("name")
        description     = _get("description")
        serial          = _get("serialNr")
        dev_type        = _get("type")
        type_display    = _get("typeDisplayName")

        # Robust ID
        try:
            device_id = int(device_id_str) if device_id_str else None
        except ValueError:
            device_id = None

        loc = parse_location_from_text(f"{name or ''} {description or ''}")

        rows.append({
            "device_id": device_id,
            "name": name,
            "description": description,
            "serialNr": serial,
            "type": dev_type,
            "typeDisplayName": type_display,
            "dnr_str": loc.get("dnr_str"),
            "feeder": loc.get("feeder"),
            "phase": loc.get("phase"),
        })

    return rows


def maybe_fetch_details(session: requests.Session, rows: List[Dict[str, Any]]) -> None:
    """Optional enrichment using /devices/{id}."""
    if not FETCH_DETAILS_PER_DEVICE:
        return
    for r in rows:
        did = r.get("device_id")
        if did is None:
            continue
        url = f"{BASE_URL}/rest/1/projects/{PROJECT}/devices/{did}"
        try:
            resp = session.get(url, timeout=TIMEOUT_S)
            resp.raise_for_status()
            r["detail_preview"] = resp.text[:300]
        except requests.RequestException as e:
            r["detail_error"] = str(e)

def main():
    ensure_dir(OUT_DIR)
    out_csv = os.path.join(OUT_DIR, CSV_FILENAME)
    out_parquet = os.path.join(OUT_DIR, PARQUET_FILENAME) if PARQUET_FILENAME else None
    raw_xml_path = os.path.join(OUT_DIR, RAW_XML_FILENAME)

    session = requests.Session()

    try:
        xml_text = fetch_devices_xml(session)
    except requests.RequestException as e:
        print(f"❌ HTTP error fetching devices list: {e}", file=sys.stderr)
        sys.exit(1)

    if SAVE_RAW_XML:
        with open(raw_xml_path, "w", encoding="utf-8") as f:
            f.write(xml_text)

    try:
        rows = parse_devices_table(xml_text)
    except Exception as e:
        print(f"❌ Parse error: {e}", file=sys.stderr)
        sys.exit(2)

    maybe_fetch_details(session, rows)

    df = pd.DataFrame(rows)
    # Column order
    cols = [
        "device_id", "name", "description", "serialNr",
        "type", "typeDisplayName", "dnr_str", "feeder", "phase"
    ]
    df = df[cols]

    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✅ Saved inventory CSV: {out_csv} (rows: {len(df)})")

    if out_parquet:
        try:
            df.to_parquet(out_parquet, index=False)
            print(f"✅ Saved inventory Parquet: {out_parquet}")
        except Exception as e:
            print(f"⚠️ Skipping Parquet save: {e}", file=sys.stderr)

    summary = {
        "rows": len(df),
        "unique_types": sorted(df["typeDisplayName"].dropna().unique().tolist()),
        "sample": df.head(3).to_dict(orient="records"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
