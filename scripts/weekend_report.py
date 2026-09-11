"""Assemble outputs/fluA/REPORT.md from whatever the weekend chain produced.

Deliberately tolerant: every section degrades to "not produced" rather than
raising, so a chain that died at stage 3 still yields a report explaining how
far it got. Reads only files on disk — it never re-runs a fit.

Inputs (all optional):
  outputs/fluA/run_status.json                 stage-by-stage result
  outputs/fluA/nuts_fluA_diagnostics.json      sampler + target summary
  outputs/fluA/compare_v5_fluA.json            v5 comparison
  outputs/fluA/policy_posterior_fluA.json      policy forward, influenza A
  outputs/eda/policy_posterior_seasonpop.json  policy forward, v5 baseline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "fluA"
ED = REPO / "outputs" / "eda"

AGES = ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]
CHANNELS = ["home", "work", "school", "other"]


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def fmt_ci(t, dec=3):
    if t is None:
        return "—"
    return f"{t[0]:.{dec}f} [{t[1]:.{dec}f}, {t[2]:.{dec}f}]"


def fmt_stat(d, dec=2, pct=True):
    if not d:
        return "—"
    u = "%" if pct else ""
    return f"{d['mean']:+.{dec}f}{u} [{d['q05']:+.{dec}f}, {d['q95']:+.{dec}f}]"


def sig(d):
    """Does the 90% interval exclude zero?"""
    if not d:
        return "—"
    return "yes" if (d["q05"] > 0) or (d["q95"] < 0) else "no"


def main() -> None:
    import os
    import sys

    # TAG keeps runs with different season sets apart; the orchestrator passes
    # the same value it used for the fit, so the report reads that run's files.
    tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("TAG", "fluA")
    suffix = "" if tag == "fluA" else f"_{tag}"

    OUT.mkdir(parents=True, exist_ok=True)
    status = jload(OUT / f"run_status{suffix}.json") or jload(OUT / "run_status.json") or {}
    diag = jload(OUT / f"nuts_{tag}_diagnostics.json")
    cmp_ = jload(OUT / f"compare_v5_{tag}.json")
    polA = jload(OUT / f"policy_posterior_{tag}.json")
    pol5 = jload(ED / "policy_posterior_seasonpop.json")

    L: list[str] = []
    a = L.append

    a("# Influenza-A refit — weekend run report")
    a("")
    a(f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
      "`scripts/weekend_report.py`. Every number below is read from a file on "
      "disk; nothing here is re-derived or estimated.")
    a("")

    # ---------------- headline ----------------------------------------------
    a("## What to read first")
    a("")
    if diag is None:
        a("The production fit did not produce a diagnostics file. Start from "
          "the stage table and the log named there.")
    else:
        g = diag.get("goal_check", {})
        ok = all(v for v in g.values() if v is not None)
        seasons = diag.get("config", {}).get("seasons", [])
        a(f"- Fitted **{len(seasons)} seasons**: {', '.join(seasons)}")
        a(f"- Convergence targets {'**all met**' if ok else '**not all met**'} "
          f"(r_hat max {diag.get('rhat_max', float('nan')):.4f}, "
          f"ESS bulk min {diag.get('ess_bulk_min', float('nan')):.0f}, "
          f"divergences {diag.get('sampling_stats', {}).get('n_divergent', '—')})")
        if cmp_ and cmp_.get("flags"):
            a("- **Flags raised in the v5 comparison — read section 3 before "
              "using any of these numbers.**")
        elif cmp_:
            a("- The v5 comparison raised no flags.")
        if polA is None:
            a("- Policy simulation did **not** complete; sections 4 and 5 are "
              "empty.")
    a("")

    # ---------------- stages -------------------------------------------------
    a("## 1. Stages")
    a("")
    if status:
        a("| stage | result | wall | log |")
        a("|---|---|---|---|")
        for st in status.get("stages", []):
            wall = st.get("wall_sec")
            wall = f"{wall / 60:.0f} min" if isinstance(wall, (int, float)) else "—"
            a(f"| {st.get('name', '?')} | {st.get('result', '?')} | {wall} | "
              f"`{st.get('log', '')}` |")
        a("")
        if status.get("aborted_at"):
            a(f"The chain stopped at **{status['aborted_at']}**. "
              f"Reason: {status.get('abort_reason', 'see the log above')}.")
            a("")
    else:
        a("No `run_status.json` — the orchestrator did not record stages.")
        a("")

    # ---------------- target -------------------------------------------------
    a("## 2. Target — what the influenza-A correction did")
    a("")
    if diag and diag.get("target"):
        t = diag["target"]
        a("| season | A-share coverage | obs A / obs A+B | population |")
        a("|---|---|---|---|")
        for s in diag["config"]["seasons"]:
            a(f"| {s} | {t['a_share_coverage'].get(s, float('nan')):.1%} | "
              f"{t['obs_A_over_all'].get(s, float('nan')):.3f} | "
              f"{t['population'].get(s, 0):,.0f} |")
        a("")
        a("Coverage is the share of fitted epidemic *mass* carrying an observed "
          "A share, not a count of weeks. `obs A / obs A+B` near 1.0 means the "
          "season was essentially pure influenza A and its target barely moved.")
    else:
        a("Not produced.")
    a("")
    a("![season diagnostic](fluA_season_diagnostic.png)")
    a("")

    # ---------------- comparison ---------------------------------------------
    a("## 3. Posterior versus the published v5 fit")
    a("")
    if cmp_:
        a("### Channel allocation")
        a("")
        a("| channel | v5 (A+B, 3 seasons) | influenza A | CI overlap |")
        a("|---|---|---|---|")
        for c in CHANNELS:
            a(f"| {c} | {fmt_ci(cmp_['v5']['pi'][c])} | "
              f"{fmt_ci(cmp_['fluA']['pi'][c])} | "
              f"{cmp_['pi_overlap'][c]:.2f} |")
        a("")
        a("`work` is the one that matters for the sick-leave conclusion — its "
          "sign drives whether sick leave helps or harms at the population "
          "level.")
        a("")
        a("### Reproduction number")
        a("")
        a("| season | v5 | influenza A | shift |")
        a("|---|---|---|---|")
        for s in cmp_["seasons"]:
            v = cmp_["v5"]["R0"].get(s)
            f = cmp_["fluA"]["R0"][s]
            sh = cmp_["r0_relative_shift"].get(s)
            a(f"| {s} | {fmt_ci(v)} | {fmt_ci(f)} | "
              f"{f'{sh:+.1%}' if sh is not None else 'new season'} |")
        a("")
        a(f"Overdispersion: v5 phi_nb {fmt_ci(cmp_['v5']['phi_nb'], 2)}, "
          f"influenza A {fmt_ci(cmp_['fluA']['phi_nb'], 2)} — "
          f"{cmp_['phi_nb_note']}.")
        a("")
        if cmp_["flags"]:
            a("### Flags")
            a("")
            for f in cmp_["flags"]:
                a(f"- **{f}**")
            a("")
        a(f"![comparison](compare_v5_{tag}.png)")
        a("")
        a(f"![fit](fit_total_{tag}.png)")
    else:
        a("Not produced.")
    a("")

    # ---------------- policy -------------------------------------------------
    a("## 4. Policy effect — total averted, by season")
    a("")
    if polA:
        seasons = [s for s in polA.get("seasons", []) if s in polA]
        a("Averted infection relative to baseline, 90% credible interval, "
          f"N = {polA.get('n_samples', '?')} posterior draws, "
          f"policy intensity p = {polA.get('policy_p', '?')}.")
        a("")
        a("| season | sick leave, term | school absence, term | "
          "sick leave, break | school absence, break | sick leave CI "
          "excludes 0 |")
        a("|---|---|---|---|---|---|")
        for s in seasons:
            d = polA[s]
            a(f"| {s} | {fmt_stat(d.get('sick_total_term'))} | "
              f"{fmt_stat(d.get('school_total_term'))} | "
              f"{fmt_stat(d.get('sick_total_vac'))} | "
              f"{fmt_stat(d.get('school_total_vac'))} | "
              f"{sig(d.get('sick_total_term'))} |")
        a("")
        if pol5:
            a("v5 baseline for the three shared seasons:")
            a("")
            a("| season | sick leave, term (v5) | school absence, term (v5) |")
            a("|---|---|---|")
            for s in [x for x in polA.get("seasons", []) if x in (pol5 or {})]:
                d = pol5[s]
                a(f"| {s} | {fmt_stat(d.get('sick_total_term'))} | "
                  f"{fmt_stat(d.get('school_total_term'))} |")
            a("")
        a("### Does the v5 conclusion survive?")
        a("")
        checks = []
        try:
            bigger = all(
                polA[s]["school_total_term"]["mean"]
                > polA[s]["sick_total_term"]["mean"] for s in seasons)
            checks.append(("School absence larger than sick leave (term)",
                           "holds" if bigger else "**REVERSED**"))
        except Exception:
            pass
        try:
            nsig = sum(1 for s in seasons
                       if sig(polA[s]["sick_total_term"]) == "no")
            checks.append((
                "Sick leave's population benefit is uncertain",
                f"{nsig} of {len(seasons)} seasons include zero"))
        except Exception:
            pass
        try:
            neg = [s for s in seasons
                   if polA[s]["sick_total_term"]["mean"] < 0]
            checks.append(("Seasons where sick leave is net harmful",
                           ", ".join(neg) if neg else "none"))
        except Exception:
            pass
        if checks:
            a("| check | result |")
            a("|---|---|")
            for k, v in checks:
                a(f"| {k} | {v} |")
            a("")
    else:
        a("Not produced.")
    a("")

    # ---------------- age ----------------------------------------------------
    a("## 5. Age redistribution under sick leave")
    a("")
    if polA:
        seasons = [s for s in polA.get("seasons", []) if s in polA]
        a("Change in attack rate by age (percentage points). A negative value "
          "means that age group gains; positive means it loses.")
        a("")
        for win, label in (("term", "school term"), ("vac", "winter break")):
            a(f"**{label}**")
            a("")
            a("| season | " + " | ".join(AGES) + " |")
            a("|---" * (len(AGES) + 1) + "|")
            for s in seasons:
                d = polA[s].get(f"sick_d_by_age_{win}", {})
                cells = []
                for ag in AGES:
                    v = d.get(ag)
                    cells.append(f"{v['mean']:+.3f}" if v else "—")
                a(f"| {s} | " + " | ".join(cells) + " |")
            a("")
    else:
        a("Not produced.")
    a("")

    # ---------------- next ---------------------------------------------------
    a("## 6. Monday")
    a("")
    a("1. Read section 3 flags first. If any fired, do not carry these numbers "
      "into the manuscript.")
    a("2. If the chain stopped early, `prompts/03_production_fit.md` and "
      "`prompts/05_downstream.md` describe how to resume — the NUTS warm-up "
      "state and the policy partial file both support resuming.")
    a("3. Remaining downstream not run by the weekend chain: age-by-metric "
      "figures, holiday reversal, R_e decomposition, the sensitivity sweeps "
      "and the theta sweep. See `prompts/05_downstream.md`.")
    a("4. Manuscript update is `prompts/06_manuscript.md`.")
    a("")

    (OUT / f"REPORT{suffix}.md").write_text("\n".join(L))
    print(f"[report] {OUT / f'REPORT{suffix}.md'}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
