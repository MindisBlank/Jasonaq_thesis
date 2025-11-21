# -------------------------------- Convenience Runner -------------------------------
# This tiny wrapper lets you avoid long CLI commands.
# Usage examples:
#   python scripts/run_janitza_report.py           # uses config/example_report.yaml
#   python scripts/run_janitza_report.py my.yaml   # custom config path
#
# The main report script remains unchanged.

# scripts/run_janitza_report.py
if False:
    pass

import sys as _sys
import os as _os
import json as _json
from pathlib import Path as _Path
try:
    import yaml as _yaml  # type: ignore
except Exception:  # graceful fallback if PyYAML is missing
    _yaml = None  # type: ignore

# Import the main entry from this module when the file is executed as a script.
from src.phasebalance.janitza_report_device import main as _report_main

_DEF_CFG = "config/example_report.yaml"

_DEF_CONTENT = {
    "device_id": 265,
    "start": "2025-11-01 00:00",
    "end": "2025-11-03 00:00",
    "timebase": "15m",
    "caps_csv": "data/capabilities.csv",
    "outdir": "results",
    "metric": "cur_ratio",
    "mild": 20,
    "moderate": 40,
    "severe": 60,
    "tau_off_ratio": 0.9,
    "gap_merge": 1,
    "min_len": 2,
    "dpi": 150,
    "no_ternary": False,
}


def _read_cfg(path: str) -> dict:
    p = _Path(path)
    if not p.exists():
        # create a starter config if it doesn't exist
        p.parent.mkdir(parents=True, exist_ok=True)
        if _yaml:
            p.write_text(_yaml.safe_dump(_DEF_CONTENT, sort_keys=False))
        else:
            p.write_text(_json.dumps(_DEF_CONTENT, indent=2))
        print(f"Created default config at {p}. Edit it and re-run.")
        return _DEF_CONTENT
    txt = p.read_text()
    if _yaml and (p.suffix.lower() in {".yml", ".yaml"}):
        return _yaml.safe_load(txt) or {}
    # fallback for .json or any text
    try:
        return _json.loads(txt)
    except Exception:
        raise SystemExit(f"Cannot parse config file: {p}. Install PyYAML or provide JSON.")


def _to_argv(cfg: dict) -> list[str]:
    # Map config keys to CLI flags expected by janitza_report_device.parse_args
    arg_map = {
        "device_id": "--device-id",
        "start": "--start",
        "end": "--end",
        "timebase": "--timebase",
        "caps_csv": "--caps-csv",
        "outdir": "--outdir",
        "metric": "--metric",
        "mild": "--mild",
        "moderate": "--moderate",
        "severe": "--severe",
        "tau_off_ratio": "--tau-off-ratio",
        "gap_merge": "--gap-merge",
        "min_len": "--min-len",
        "dpi": "--dpi",
        "no_ternary": "--no-ternary",
    }
    argv: list[str] = []
    for k, flag in arg_map.items():
        if k not in cfg:
            continue
        v = cfg[k]
        if isinstance(v, bool):
            if v:  # only include true booleans (e.g., --no-ternary)
                argv.append(flag)
        else:
            argv += [flag, str(v)]
    return argv


def run_from_config(path: str | None = None) -> int:
    cfg_path = path or _os.environ.get("JANITZA_REPORT_CFG", _DEF_CFG)
    cfg = _read_cfg(cfg_path)
    argv = _to_argv(cfg)
    return _report_main(argv)


if __name__ == "__main__":
    cfg = _sys.argv[1] if len(_sys.argv) > 1 else None
    raise SystemExit(run_from_config(cfg))
