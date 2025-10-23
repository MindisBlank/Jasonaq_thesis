import pandas as pd
from janitza_fetch import fetch_hist_json
import time
import math



def _series_from_values(obj, label):
    """
    Turn {"values":[{"startTime":ns, "avg":x}, ...]} into a pandas Series
    indexed by startTime (ns) with the 'avg' value. label is only for naming.
    """
    vals = obj.get("values", []) if isinstance(obj, dict) else []
    data = {}
    for v in vals:
        st = v.get("startTime")
        avg = v.get("avg")
        if st is None or avg is None:
            continue
        try:
            data[int(st)] = float(avg)
        except (TypeError, ValueError):
            continue
    if not data:
        return pd.Series(dtype=float, name=label)
    s = pd.Series(data).sort_index()
    s.name = label
    return s

def _compute_stats(json_in1, json_in4):
    """
    Align Input01 & Input04 series by startTime and compute:
      - mean absolute difference (MAD) of avg values over the overlap
      - ratio = sum(avg_in1) / sum(avg_in4) over the overlap
    Returns (mad, ratio, n_points) or (None, None, 0) if insufficient overlap.
    """
    s1 = _series_from_values(json_in1, "in1")
    s4 = _series_from_values(json_in4, "in4")

    if s1.empty or s4.empty:
        return (None, None, 0)

    joined = pd.concat([s1, s4], axis=1, join="inner").dropna()
    if joined.empty:
        return (None, None, 0)

    mad = (joined["in1"] - joined["in4"]).abs().mean()

    denom = joined["in4"].sum()
    ratio = (joined["in1"].sum() / denom) if abs(denom) > 1e-12 else None

    return (float(mad), (float(ratio) if ratio is not None else None), int(len(joined)))


def has_capability(cap_df, device_id, **criteria):
    """
    Return True/False if the given device_id has at least one row
    in cap_df matching all non-None criteria.
    Example:
        has_capability(cap, 301, value_backend="I_Effective", type_backend="Input01")
    """
    #This is really slow condsider making it better later
    # Normalize device_id column to string for safe comparison
    cap_df = cap_df.copy()
    cap_df["device_id"] = cap_df["device_id"].astype(str)
    device_id = str(device_id)

    df = cap_df[cap_df["device_id"] == device_id]
    if df.empty:
        return False

    # Apply all criteria (ignore None)
    for col, val in criteria.items():
        if val is None or col not in df.columns:
            continue
        df = df[df[col].astype(str) == str(val)]
        if df.empty:
            return False

    return not df.empty


def main():
    devices = pd.read_csv("metadata/devices.csv")
    caps = pd.read_csv("metadata/capabilities.csv")
    SampleTime="15m"
    start="2025-09-01 12:00"
    end="2025-10-01 12:00"





    # Normalize device_id in both
    devices["device_id"] = devices["device_id"].astype(str)
    caps["device_id"] = caps["device_id"].astype(str)

    matches = []
    for _, row in devices.iterrows():
        did = row["device_id"]
        name = row["name"]

        # Check both conditions
        has_input1 = has_capability(caps, did, value_backend="I_Effective", type_backend="Input01")
        has_input4 = has_capability(caps, did, value_backend="I_Effective", type_backend="Input04")

        if has_input1 and has_input4:
            matches.append((did, name))

    # Print results
    print(f"\nTotal matches: {len(matches)}")


    # Fetch data for each valid device
    results = []  # will hold tuples: (device_id, name, mad, ratio, n_points)
    for did, name in matches:
        print(f"📡 Fetching data for device {did} ({name})...")

        data_input01 = fetch_hist_json(
                device_id=did,
                variable_backend="I_Effective",
                phase_backend="Input01",
                timebase=SampleTime,
                start=start,
                end=end,
            )

        data_input04 = fetch_hist_json(
                device_id=did,
                variable_backend="I_Effective",
                phase_backend="Input04",
                timebase=SampleTime,
                start=start,
                end=end,
            )
        
        mad, ratio, n_points = _compute_stats(data_input01, data_input04)

        if mad is None or ratio is None or n_points == 0:
            print(f"   ⚠️  No overlapping data to compare for device {did} ({name}).")
        else:
            results.append((did, name, mad, ratio, n_points))
            print(f"   ➜ overlap={n_points} points | MAD={mad:.6f} | ratio={ratio:.6f}")

        print(f"✅ Got data for device {did} ({name}) — lengths: "
                  f"{len(data_input01.get('values', []))} vs {len(data_input04.get('values', []))}")

        # small pause to avoid hammering the server
        time.sleep(10)

    # Keep only devices with valid stats
    valid = [r for r in results if r[2] is not None and r[3] is not None and r[4] > 0]
    if not valid:
        print("\nNo comparable devices with overlapping data.")
    else:
        # Sort by smallest MAD (difference) ascending
        valid.sort(key=lambda x: x[2])

        top3 = valid[:3]
        print("\nTop 3 substations (devices) with smallest difference between Input01 and Input04")
        print("device_id,name,ratio")
        for did, name, mad, ratio, n_points in top3:
            print(f"{did},{name},{ratio:.6f}")

if __name__ == "__main__":
    main()
