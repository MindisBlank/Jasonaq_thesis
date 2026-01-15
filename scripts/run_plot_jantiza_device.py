import yaml
import pandas as pd
import sys as _sys
from pathlib import Path as _Path, Path
# Import your original module (assuming it is named janitza_tool.py)
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))
from src.phasebalance.plot_janitza import fetch_multidevice_variable, plot_multidevice



def run_yaml_config(config_path="configs/plot_janitza.yaml"):
    # 1. Load the YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 2. Separate config for fetching vs general logic
    # We pop keys that aren't arguments for fetch_multidevice_variable
    parquet_path = config.pop("parquet_output", None)
    no_plot = config.pop("no_plot", False)

    # 3. Call the fetch function using dictionary unpacking
    print("Fetching data...")
    df = fetch_multidevice_variable(**config)

    if df.empty:
        print("No data returned.")
        return

    # 4. Save to Parquet if requested
    if parquet_path:
        path = Path(parquet_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        print(f"Saved to {path}")

    # 5. Plot if not disabled
    if not no_plot:
        plot_multidevice(
            df,
            device_ids=config["device_ids"],
            variable_backend=config["variable_backend"],
            which=config["which"],
            start=config["start"],
            end=config["end"],
            show=True
        )

if __name__ == "__main__":
    run_yaml_config()