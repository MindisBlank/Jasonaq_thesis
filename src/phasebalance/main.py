# main.py
from janitza_fetch import fetch_hist_json
from plot_janitza import simple_line_plot  # or line_with_band_plot
import matplotlib.pyplot as plt

def main():
    data = fetch_hist_json(
        device_id=408,
        variable_backend="I_Effective",
        phase_backend="Input04",
        timebase="15m",
        start="2025-09-01 12:00",
        end="2025-10-02 12:00",
        dry_run=False,
        auth_token=None,
    )

    #print("raw data:", data)

    # Simple line of the average values
    fig = simple_line_plot(data, which="avg", title="Current effective (avg)")
    if fig:
        plt.show()

if __name__ == "__main__":
    main()
