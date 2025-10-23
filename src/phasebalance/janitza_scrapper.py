import pandas as pd




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
    print("device_id,name")
    for did, name in matches:
        print(f"{did},{name}")
    print(f"\nTotal matches: {len(matches)}")


if __name__ == "__main__":
    main()
