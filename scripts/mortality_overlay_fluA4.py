"""T4 (prompts/10) -- illustrative mortality overlay for the fluA4 policy runs.

main_v5's Discussion states "about two-thirds of school absence's averted
deaths are among the 65+", from a calculation whose script is not in the
repository. This rebuilds that calculation from
outputs/fluA/policy_posterior_fluA4.json's per-age averted-infection counts
({sick,school}_num_by_age_{term,vac}), weighted by an age-specific IFR.

The exact v5 IFR vector is not recoverable, so this defines a new one at the
order-of-magnitude v5 describes (children ~1e-5, elderly ~1e-2; basis:
McDonald2023) and records it verbatim in the output JSON rather than
asserting precision. Per the prompt, this is explicitly "illustrative only":
no claim of point-estimate accuracy, and a shape-sensitivity check (three IFR
curves spanning a 10x range in how steeply IFR rises with age) is reported
alongside the headline numbers so a reader can see how much the 65+ share
depends on the assumed curve.

No simulation -- this is a pure post-processing pass over an existing JSON,
so it runs in seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUTD = REPO / "outputs" / "fluA"
POLICY_JSON = OUTD / "policy_posterior_fluA4.json"

HIRA_AGES = ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]

#: New order-of-magnitude IFR vectors (not the v5 vector -- see docstring).
#: "main" spans children ~1e-5 to elderly ~1e-2, matching main_v5's stated
#: order of magnitude. "shallow"/"steep" vary how compressed/spread the
#: age gradient is, holding the elderly value fixed, to test how much the
#: 65+ death share depends on that shape rather than on absolute IFR level.
IFR_CURVES = {
    "main":    {"0-5": 1e-5, "6-11": 1e-5, "12-17": 2e-5,
                "18-44": 1e-4, "45-64": 1e-3, "65+": 1e-2},
    "shallow": {"0-5": 1e-4, "6-11": 1e-4, "12-17": 2e-4,
                "18-44": 5e-4, "45-64": 2e-3, "65+": 1e-2},
    "steep":   {"0-5": 1e-6, "6-11": 1e-6, "12-17": 2e-6,
                "18-44": 2e-5, "45-64": 5e-4, "65+": 1e-2},
}
IFR_BASIS = "McDonald2023 (order-of-magnitude placeholder, exact v5 vector not in repo)"


def overlay_one(num_by_age: dict, ifr: dict) -> dict:
    """averted deaths per age (mean-only -- see module docstring on why
    quantiles are not error-propagated) + 65+ share of the total."""
    deaths = {ag: ifr[ag] * num_by_age[ag]["mean"] for ag in HIRA_AGES}
    total = sum(deaths.values())
    share_65p = (deaths["65+"] / total) if total != 0 else None
    return dict(averted_deaths_by_age=deaths, total_averted_deaths=total,
                share_65p=share_65p)


def main():
    if not POLICY_JSON.exists():
        raise FileNotFoundError(f"{POLICY_JSON} missing")
    d = json.loads(POLICY_JSON.read_text())
    seasons = d["seasons"]

    out = dict(source=str(POLICY_JSON.relative_to(REPO)),
                ifr_curves=IFR_CURVES, ifr_basis=IFR_BASIS,
                seasons=seasons, results={})

    sign_check = []   # sick-leave 65+ averted-death sign per (season, window)
    print("=" * 90)
    print("T4 -- mortality overlay (illustrative only)")
    print("=" * 90)
    for s in seasons:
        sd = d[s]
        out["results"][s] = {}
        for policy in ("sick", "school"):
            for win in ("term", "vac"):
                num_by_age = sd[f"{policy}_num_by_age_{win}"]
                curves_out = {}
                for curve_name, ifr in IFR_CURVES.items():
                    curves_out[curve_name] = overlay_one(num_by_age, ifr)
                out["results"][s][f"{policy}_{win}"] = curves_out
                main_res = curves_out["main"]
                print(f"  [{s}][{policy}/{win}] main IFR: total averted deaths="
                      f"{main_res['total_averted_deaths']:+.2f}  "
                      f"65+ share={main_res['share_65p']}")
                if policy == "sick":
                    raw_65p_mean = num_by_age["65+"]["mean"]
                    sign_check.append(dict(season=s, window=win,
                                             sick_65p_averted_infections_mean=raw_65p_mean,
                                             sign="positive" if raw_65p_mean > 0 else
                                                  ("negative" if raw_65p_mean < 0 else "zero")))

    out["sick_leave_65p_sign_check"] = sign_check
    n_pos = sum(1 for r in sign_check if r["sign"] == "positive")
    print(f"\n[sign check] sick-leave 65+ averted infections positive in "
          f"{n_pos}/{len(sign_check)} (season, window) combos "
          f"(v5 Discussion described this as ~0 or slightly negative)")
    for r in sign_check:
        print(f"    {r['season']} / {r['window']}: mean={r['sick_65p_averted_infections_mean']:+.2f} "
              f"({r['sign']})")

    out_path = OUTD / "mortality_overlay_fluA4.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(f"\n[json] {out_path}")


if __name__ == "__main__":
    main()
