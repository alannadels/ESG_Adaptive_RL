"""Plot the walk-forward regime path: price shaded by regime + equity curves."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

COLORS = {"S1_calm": "#2ca02c", "S2_choppy": "#ff7f0e", "S3_stress": "#d62728"}


def plot(source: str) -> str:
    df = pd.read_csv(os.path.join(RESULTS, f"{source}_walkforward.csv"),
                     parse_dates=["date"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})
    ax1.plot(df["date"], df["close"], color="black", lw=0.8, zorder=3)
    start, prev = df["date"].iloc[0], None
    for i in range(len(df)):
        r = df["regime"].iloc[i]
        if prev is not None and r != prev:
            ax1.axvspan(start, df["date"].iloc[i], color=COLORS.get(prev, "gray"),
                        alpha=0.18, lw=0)
            start = df["date"].iloc[i]
        prev = r
    ax1.axvspan(start, df["date"].iloc[-1], color=COLORS.get(prev, "gray"), alpha=0.18, lw=0)
    ax1.set_title(f"{source} — ESG regime HMM (walk-forward OOS): price shaded by regime",
                  fontsize=13)
    ax1.set_ylabel("Index level / price")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in COLORS.values()]
    ax1.legend(handles, ["Calm (S1)", "Choppy (S2)", "Stress (S3)"], loc="upper left", ncol=3)
    ax1.grid(alpha=0.25)

    ax2.plot(df["date"], df["bh_equity"], label="Buy & Hold", color="#1f77b4", lw=1.3)
    ax2.plot(df["date"], df["strat_equity"], label="Regime Overlay (100/60/0)",
             color="#d62728", lw=1.3)
    ax2.set_ylabel("Growth of $1"); ax2.set_yscale("log")
    ax2.legend(loc="upper left"); ax2.grid(alpha=0.25)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    out = os.path.join(RESULTS, f"{source}_regimes.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["esg_index", "SUSA", "DSI"]):
        print("saved", plot(s))
