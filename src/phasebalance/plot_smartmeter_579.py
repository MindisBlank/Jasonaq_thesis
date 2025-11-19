import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_smartmeter_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Ensure ts is datetime and sorted
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")

    return df


def main():
    path = Path("data/smartmeter_15min_579_20251101_20251107.parquet")
    df = load_smartmeter_parquet(path)

    print(df.head())
    print(df.columns)

    # If the file ever contains multiple substations, filter here
    # df = df[df["substation_id"] == 579]

    # --- 1) Plot total current vs time ---
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["ts"], df["I_total"], label="I_total")
    ax.set_title("Substation 579 – Total current (15-min)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Current [A]")
    ax.grid(True, alpha=0.3)
    ax.legend()
 
    # --- 2) Plot phase currents vs time ---
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(df["ts"], df["I_a"], label="I_a (L1)")
    ax2.plot(df["ts"], df["I_b"], label="I_b (L2)")
    ax2.plot(df["ts"], df["I_c"], label="I_c (L3)")
    ax2.set_title("Substation 579 – Phase currents (15-min)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Current [A]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # --- 3) Optional: n_mps over time (sanity check) ---
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.step(df["ts"], df["n_mps"], where="post")
    ax3.set_title("Substation 579 – number of metering points contributing")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("n_mps")
    ax3.grid(True, alpha=0.3)

    # Optional: print per_completed (should be constant per substation)
    print("Unique per_completed values:", df["per_completed"].unique())

    plt.show()


if __name__ == "__main__":
    main()
