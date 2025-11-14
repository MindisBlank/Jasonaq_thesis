"""Convenience wrapper for src.phasebalance.janitza_plot_multi_phase.

This script mirrors the behaviour of running janitza_plot_multi_phase.py from
```
python -m src.phasebalance.janitza_plot_multi_phase
```
but lets you manage the long list of options via a lightweight config file.

Usage
-----
python scripts/run_janitza_multi_phase.py           # uses configs/example_multi_phase.yaml
python scripts/run_janitza_multi_phase.py my_cfg.yaml

Set JANITZA_MULTI_PHASE_CFG to point to an alternate config when you do not
want to pass a CLI argument each time.
"""

from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path
import sys as _sys
from typing import Any as _Any, Dict as _Dict

try:
    import yaml as _yaml  # type: ignore
except Exception:  # pragma: no cover - PyYAML is optional
    _yaml = None  # type: ignore

_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
    
from src.phasebalance.janitza_plot_multi_phase import plot_substation_measurement as _plot_substation

_DEF_CFG = "configs/example_multi_phase.yaml"

_DEF_CONTENT: _Dict[str, _Any] = {
    "dnr_str": "D0579",
    "value_backend": "I_Effective",
    "profile": "3phase_I",
    "devices_csv": "metadata/devices.csv",
    "capabilities_csv": "metadata/capabilities.csv",
    "timebase": "15m",
    "start": "2025-09-01 00:00",
    "end": "2025-10-02 00:00",
    "which": "avg",
    "combine": "sum",
    "auth_token": None,
    "show": True,
}


def _read_cfg(path: str) -> _Dict[str, _Any]:
    cfg_path = _Path(path)
    if not cfg_path.exists():
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        if _yaml and cfg_path.suffix.lower() in {".yml", ".yaml"}:
            cfg_path.write_text(_yaml.safe_dump(_DEF_CONTENT, sort_keys=False))
        else:
            cfg_path.write_text(_json.dumps(_DEF_CONTENT, indent=2))
        print(f"Created default config at {cfg_path}. Edit it and re-run.")
        return dict(_DEF_CONTENT)

    text = cfg_path.read_text()
    if _yaml and cfg_path.suffix.lower() in {".yml", ".yaml"}:
        data = _yaml.safe_load(text) or {}
    else:
        data = _json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {cfg_path} must contain a mapping/dict.")
    return data  # type: ignore[return-value]


def _normalize_bool(value: _Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in {"1", "true", "yes", "on"}:
            return True
        if value.lower() in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"Cannot interpret boolean value from {value!r}.")


def _to_kwargs(cfg: _Dict[str, _Any]) -> _Dict[str, _Any]:
    mapping = {
        "dnr_str": str,
        "value_backend": str,
        "profile": str,
        "timebase": str,
        "start": str,
        "end": str,
        "which": str,
        "combine": str,
        "auth_token": (lambda v: None if v in {None, "", "null"} else str(v)),
        "show": _normalize_bool,
    }

    kwargs: _Dict[str, _Any] = {}
    for key, caster in mapping.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if callable(caster):
            kwargs[key] = caster(value)
        else:  # pragma: no cover - defensive
            kwargs[key] = caster(value)

    if "devices_csv" in cfg:
        kwargs["devices_csv"] = _Path(cfg["devices_csv"])
    if cfg.get("capabilities_csv"):
        kwargs["capabilities_csv"] = _Path(cfg["capabilities_csv"])
    else:
        kwargs["capabilities_csv"] = None

    return kwargs


def run_from_config(path: str | None = None) -> int:
    cfg_path = path or _os.environ.get("JANITZA_MULTI_PHASE_CFG", _DEF_CFG)
    config = _read_cfg(cfg_path)
    try:
        kwargs = _to_kwargs(config)
    except ValueError as exc:  # failed to parse boolean etc.
        print(f"❌ {exc}")
        return 1

    try:
        _plot_substation(**kwargs)
    except Exception as exc:  # pragma: no cover - CLI wrapper
        print(f"❌ Failed to run janitza_plot_multi_phase: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    cfg_arg = _sys.argv[1] if len(_sys.argv) > 1 else None
    raise SystemExit(run_from_config(cfg_arg))
