"""Figure fig:luminance (Chapter 3): what the screen shows for the slowest and the
fastest target during the first 0.3 s of the flicker.

Run from the repository root:
    .venv/bin/python thesis/figures/make_luminance.py
Writes thesis/figures/luminance.pdf.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ssvep import config, stimulus

OUT = Path(__file__).with_name("luminance.pdf")
DURATION = 0.3  # s after the onset
R = config.REFRESH_RATE
FREQS = [config.TARGET_FREQS[0], config.TARGET_FREQS[-1]]  # 8.0 and 15.0 Hz

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(2, 1, figsize=(5.9, 3.4), sharex=True)

frames = np.arange(int(round(DURATION * R)) + 1)  # refreshes after the onset
fine = np.linspace(0, frames[-1], 2000)  # the same time axis, in refreshes
for ax, freq in zip(axes, FREQS):
    smooth = stimulus.luminance([freq], [0.0], fine, R)[0]
    shown = stimulus.gray_levels(stimulus.luminance([freq], [0.0], frames, R))[0] / 255
    ax.plot(1000 * fine / R, smooth, color="0.65", lw=1, label="sinusoid")
    ax.step(1000 * frames / R, shown, where="post", color="black", lw=1,
            label="gray level on each refresh")
    ax.plot(1000 * frames / R, shown, "o", color="black", ms=2.5)
    ax.set_title(f"{freq:.1f} Hz ({R / freq:.2f} refreshes per cycle)",
                 loc="left", fontsize=9)
    ax.set_ylabel("Luminance")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 0.5, 1])
axes[0].legend(loc="upper right", frameon=False, fontsize=8, ncol=2,
               bbox_to_anchor=(1.0, 1.32))
axes[-1].set_xlabel("Time after the onset (ms)")
axes[-1].set_xlim(0, 1000 * DURATION)
fig.tight_layout()
fig.savefig(OUT)
print(f"Saved {OUT}")
