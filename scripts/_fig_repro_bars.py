"""(A) 채널별 R_c / (B) 연령별 R_b 막대 (reproduction_numbers_2019-2020.json, static)."""
import os, json, sys
from pathlib import Path
import numpy as np
os.environ["FIG_OUTDIR_SUFFIX"] = "_seasonpop"
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from figstyle import savefig, COL_SCHOOL, COL_ZERO, W_DOUBLE

D = json.load(open(REPO / "outputs/eda/reproduction_numbers_2019-2020.json"))
st = D["static"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.2), constrained_layout=True)

# ---- (A) channel R_c vs pi share ----
chans = ["home", "work", "school", "other"]
pi_share = {"home": 44.7, "work": 29.2, "school": 6.4, "other": 19.7}
rc = [st["R_c_share_pct"][c]["mean"] for c in chans]
rc_lo = [st["R_c_share_pct"][c]["mean"] - st["R_c_share_pct"][c]["q05"] for c in chans]
rc_hi = [st["R_c_share_pct"][c]["q95"] - st["R_c_share_pct"][c]["mean"] for c in chans]
ps = [pi_share[c] for c in chans]
x = np.arange(len(chans)); w = 0.38
axA.bar(x - w/2, ps, w, color="#C7C7C7", label=r"transmissibility share $\pi_c$")
axA.bar(x + w/2, rc, w, yerr=[rc_lo, rc_hi], capsize=3,
        color=COL_SCHOOL, label=r"reproduction share $R_c/R_0$")
for i, c in enumerate(chans):
    if c == "home":
        continue  # ratio 1.0x is self-evident and overlaps the legend
    ratio = rc[i] / ps[i]
    axA.text(x[i] + w/2, rc[i] + rc_hi[i] + 1.2, f"{ratio:.1f}$\\times$",
             ha="center", fontsize=8, fontweight="bold", color=COL_SCHOOL)
axA.set_xticks(x); axA.set_xticklabels(chans)
axA.set_ylabel("Share of total (%)")
axA.set_title(r"(A) Channel contribution to $R_0$ vs share",
              fontsize=9.5, fontweight="bold")
axA.legend(frameon=False, fontsize=7.5, loc="upper right")
axA.set_ylim(0, max(max(ps), max(rc)) * 1.25)

# ---- (B) age R_b ----
ages = D["static"]["age_labels_6"]
rb = [st["R_b_6"][a]["mean"] for a in ages]
rb_lo = [st["R_b_6"][a]["mean"] - st["R_b_6"][a]["q05"] for a in ages]
rb_hi = [st["R_b_6"][a]["q95"] - st["R_b_6"][a]["mean"] for a in ages]
shr = [st["R_b_6_share_pct"][a]["mean"] for a in ages]
# children 0-17 highlighted
cols = [COL_SCHOOL if a in ("0-5", "6-11", "12-17") else "#B0B0B0" for a in ages]
xb = np.arange(len(ages))
axB.bar(xb, rb, 0.62, yerr=[rb_lo, rb_hi], capsize=3, color=cols)
for i, a in enumerate(ages):
    axB.text(xb[i], rb[i] + rb_hi[i] + 0.03, f"{shr[i]:.0f}%",
             ha="center", fontsize=8, color=cols[i], fontweight="bold")
axB.set_xticks(xb); axB.set_xticklabels(ages, rotation=0, fontsize=8)
axB.set_ylabel(r"Secondary infections generated $R_b$")
child = sum(shr[:3])
axB.set_title(rf"(B) Age contribution (children 0--17: {child:.0f}%)",
              fontsize=9.5, fontweight="bold")
axB.set_ylim(0, max(rb[i] + rb_hi[i] for i in range(len(ages))) * 1.18)

savefig(fig, "repro_bars")
print(f"saved repro_bars; children share = {child:.0f}%, school R/pi = {rc[2]/ps[2]:.1f}x")
