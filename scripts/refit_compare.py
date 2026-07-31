"""S4: old (v4, 2023 pop) vs new (seasonpop) 비교 표 + 해석 체크.

읽는 파일:
  old NUTS:   outputs/eda/nuts_v4_merged_diagnostics.json
  new NUTS:   outputs/eda/nuts_seasonpop_merged_diagnostics.json
  old policy: outputs/eda/policy_posterior_v4.json         (term window only)
  new policy: outputs/eda/policy_posterior_seasonpop.json  (term + vac)

산출:
  outputs/eda/refit_compare_old_vs_new.json
  logs/refit_compare.md
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
ED = REPO / "outputs" / "eda"
LOG = REPO / "logs" / "refit_compare.md"
OUT_JSON = ED / "refit_compare_old_vs_new.json"

OLD_NUTS = ED / "nuts_v4_merged_diagnostics.json"
NEW_NUTS = ED / "nuts_seasonpop_merged_diagnostics.json"
OLD_POL  = ED / "policy_posterior_v4.json"
NEW_POL  = ED / "policy_posterior_seasonpop.json"

SEAS = ["2016-2017", "2017-2018", "2019-2020"]
CH = ["home", "work", "school", "other"]
AGES = ["0-5", "6-11", "12-17", "18-44", "45-64", "65+"]
CHILD = ["0-5", "6-11", "12-17"]
ADULT = ["18-44", "45-64"]


def _fmt(x, prec=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return f"{x:{prec+2}.{prec}f}"


def _pct(x, prec=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  n/a"
    return f"{x:+.{prec}f}%"


def _ci_contains_zero(ci):
    if not ci: return None
    q05 = ci.get("q05"); q95 = ci.get("q95")
    if q05 is None or q95 is None: return None
    return (q05 <= 0.0 <= q95)


def _load(p):
    if not p.exists(): return None
    return json.loads(p.read_text())


def _old_nuts_pi():
    """v4 병합 진단에는 pi_work_ci 만 저장됨. 나머지 채널은 raw npz 로부터 재계산 필요.
    간단히: 두 파일 없이 skip 처리."""
    d = _load(OLD_NUTS)
    if not d: return None
    out = {}
    # merged 섹션에서 pi[k] mean, hdi 추출
    m = d.get("merged", {})
    for k in range(4):
        key = f"pi[{k}]"
        if key in m:
            e = m[key]
            out[CH[k]] = dict(mean=e.get("mean"), q025=e.get("hdi_3%"),
                               q975=e.get("hdi_97%"))
    return out


def _new_nuts_pi():
    d = _load(NEW_NUTS)
    if not d: return None
    return d.get("pi", {})


def _old_nuts_R0():
    d = _load(OLD_NUTS)
    if not d: return None
    out = {}
    r0 = d.get("R0_ci", {})
    for s in SEAS:
        if s in r0 and "merged" in r0[s]:
            e = r0[s]["merged"]
            out[s] = dict(mean=e.get("mean"), q025=e.get("q025"), q975=e.get("q975"))
    return out


def _new_nuts_R0():
    d = _load(NEW_NUTS)
    if not d: return None
    return d.get("R0", {})


def main():
    old_nuts_pi = _old_nuts_pi(); new_nuts_pi = _new_nuts_pi()
    old_nuts_R0 = _old_nuts_R0(); new_nuts_R0 = _new_nuts_R0()
    old_pol = _load(OLD_POL); new_pol = _load(NEW_POL)

    def _pi_row(ch):
        o = (old_nuts_pi or {}).get(ch, {})
        n = (new_nuts_pi or {}).get(ch, {})
        return dict(old_mean=o.get("mean"), old_lo=o.get("q025"), old_hi=o.get("q975"),
                    new_mean=n.get("mean"), new_lo=n.get("q025"), new_hi=n.get("q975"),
                    delta=(n.get("mean") - o.get("mean")) if
                          (o.get("mean") is not None and n.get("mean") is not None) else None)

    def _R0_row(s):
        o = (old_nuts_R0 or {}).get(s, {})
        n = (new_nuts_R0 or {}).get(s, {})
        return dict(old_mean=o.get("mean"), old_lo=o.get("q025"), old_hi=o.get("q975"),
                    new_mean=n.get("mean"), new_lo=n.get("q025"), new_hi=n.get("q975"),
                    delta_pct=(100 * (n.get("mean") - o.get("mean")) / o.get("mean"))
                              if (o.get("mean") not in (None, 0) and n.get("mean") is not None)
                              else None)

    compare = dict(
        pi={ch: _pi_row(ch) for ch in CH},
        R0={s: _R0_row(s) for s in SEAS},
        seasons={s: {} for s in SEAS},
    )

    # ── Averted totals per season ──
    for s in SEAS:
        o = (old_pol or {}).get(s, {}); n = (new_pol or {}).get(s, {})
        # OLD schema: sick_total, school_total  (term window only)
        # NEW schema: sick_total_term, sick_total_vac, ...
        old_sick = o.get("sick_total", {}); old_school = o.get("school_total", {})
        new_sick_t = n.get("sick_total_term", {}); new_sick_v = n.get("sick_total_vac", {})
        new_school_t = n.get("school_total_term", {}); new_school_v = n.get("school_total_vac", {})

        row = dict(
            old_sick_term=dict(mean=old_sick.get("mean"),
                                q05=old_sick.get("q05"), q95=old_sick.get("q95"),
                                zero_in_ci=_ci_contains_zero(old_sick)),
            new_sick_term=dict(mean=new_sick_t.get("mean"),
                                q05=new_sick_t.get("q05"), q95=new_sick_t.get("q95"),
                                zero_in_ci=_ci_contains_zero(new_sick_t)),
            new_sick_vac=dict(mean=new_sick_v.get("mean"),
                               q05=new_sick_v.get("q05"), q95=new_sick_v.get("q95"),
                               zero_in_ci=_ci_contains_zero(new_sick_v)),
            old_school_term=dict(mean=old_school.get("mean"),
                                  q05=old_school.get("q05"), q95=old_school.get("q95"),
                                  zero_in_ci=_ci_contains_zero(old_school)),
            new_school_term=dict(mean=new_school_t.get("mean"),
                                  q05=new_school_t.get("q05"), q95=new_school_t.get("q95"),
                                  zero_in_ci=_ci_contains_zero(new_school_t)),
            new_school_vac=dict(mean=new_school_v.get("mean"),
                                 q05=new_school_v.get("q05"), q95=new_school_v.get("q95"),
                                 zero_in_ci=_ci_contains_zero(new_school_v)),
        )
        # per-age Δattack (compare sick_by_age term)
        old_by = o.get("sick_by_age", {}); new_by = n.get("sick_d_by_age_term", {})
        row["sick_d_by_age_term"] = {}
        for a in AGES:
            oa = old_by.get(a, {}); na = new_by.get(a, {})
            row["sick_d_by_age_term"][a] = dict(
                old_mean=oa.get("mean"), old_q05=oa.get("q05"), old_q95=oa.get("q95"),
                old_sig=(_ci_contains_zero(oa) == False if oa else None),
                new_mean=na.get("mean"), new_q05=na.get("q05"), new_q95=na.get("q95"),
                new_sig=(_ci_contains_zero(na) == False if na else None),
                sign_flip=((oa.get("mean") or 0) * (na.get("mean") or 0) < 0)
                          if (oa.get("mean") is not None and na.get("mean") is not None)
                          else None,
            )
        # absolute counts (adult sick / child school / net / school policy)
        old_num_by = o.get("sick_num_by_age", {}); new_num_by = n.get("sick_num_by_age_term", {})
        def _sum_ages(d, ages):
            out = {}
            for stat in ("mean","q05","q95"):
                out[stat] = sum(d.get(a, {}).get(stat) or 0.0 for a in ages)
            return out
        row["sick_num_adult_term"] = dict(old=_sum_ages(old_num_by, ADULT),
                                            new=(n.get("sick_num_adult_term", {})))
        row["sick_num_child_term"] = dict(old=_sum_ages(old_num_by, CHILD),
                                            new=(n.get("sick_num_child_term", {})))
        row["sick_num_net_term"] = dict(old=_sum_ages(old_num_by, AGES),
                                          new=(n.get("sick_num_net_term", {})))
        row["school_num_child_term"] = dict(old=_sum_ages(o.get("school_num_by_age", {}), CHILD),
                                              new=(n.get("school_num_child_term", {})))
        # child weighted attack (only new)
        row["sick_child_d_term"] = n.get("sick_child_d_term")
        row["school_child_d_term"] = n.get("school_child_d_term")
        row["baseline_total"] = dict(new=n.get("baseline_total"))
        row["child_baseline_attack"] = dict(new=n.get("child_baseline_attack"))
        compare["seasons"][s] = row

    # ── 해석 체크 ──
    check = {}
    # (a) 시즌별 병가 0∈CI 변화?
    zero_check = {}
    for s in SEAS:
        r = compare["seasons"][s]
        old_zero = r.get("old_sick_term", {}).get("zero_in_ci")
        new_zero = r.get("new_sick_term", {}).get("zero_in_ci")
        zero_check[s] = dict(old=old_zero, new=new_zero, changed=(old_zero != new_zero))
    check["sick_zero_in_ci"] = zero_check

    # (b) 성인↓/아동↑ 방향 3/3 유지 (병가 term)?
    ac_dir = {}
    all_3 = True
    for s in SEAS:
        by = compare["seasons"][s].get("sick_d_by_age_term", {})
        adult_signs = [np.sign(by.get(a, {}).get("new_mean") or 0) for a in ADULT]
        child_signs = [np.sign(by.get(a, {}).get("new_mean") or 0) for a in CHILD]
        adult_neg = all(s == -1 for s in adult_signs)  # d_attack < 0 → 감염 감소
        child_pos = all(s == +1 for s in child_signs)  # d_attack > 0 → 감염 증가 (파급)
        holds = adult_neg and child_pos
        ac_dir[s] = dict(adult_neg=adult_neg, child_pos=child_pos, holds=holds)
        if not holds: all_3 = False
    check["adult_down_child_up_3of3"] = dict(by_season=ac_dir, all_hold=all_3)

    # (c) 학교 > 병가 유지?
    sc_vs = {}
    for s in SEAS:
        r = compare["seasons"][s]
        n_school = r.get("new_school_term", {}).get("mean")
        n_sick = r.get("new_sick_term", {}).get("mean")
        sc_vs[s] = dict(school_gt_sick=(n_school is not None and n_sick is not None
                                          and n_school > n_sick),
                        school=n_school, sick=n_sick)
    check["school_gt_sick"] = sc_vs

    # (d) 시즌 간 재분배 크기 순서 (child_d_term abs 정렬) 변화?
    child_reord = {}
    if all(new_pol.get(s, {}).get("sick_child_d_term") for s in SEAS):
        arr = [(s, abs(new_pol[s]["sick_child_d_term"]["mean"])) for s in SEAS]
        order_new = [x[0] for x in sorted(arr, key=lambda t: -t[1])]
        child_reord = dict(order_new=order_new)
    check["seasonal_child_reorder"] = child_reord

    # ── Flag CI-boundary / sign flip ──
    flags = []
    for s in SEAS:
        r = compare["seasons"][s]
        # sick_total_term CI crossover
        z = zero_check[s]
        if z.get("changed"):
            flags.append(f"[{s}] sick_total_term 0∈CI 변화: old={z['old']} new={z['new']}")
        # per-age sign flip
        for a in AGES:
            e = r["sick_d_by_age_term"].get(a, {})
            if e.get("sign_flip"):
                flags.append(f"[{s}] sick_d[{a}] 부호 변경 "
                              f"old={_pct(e['old_mean'])} new={_pct(e['new_mean'])}")

    compare["checks"] = check
    compare["flags"] = flags

    OUT_JSON.write_text(json.dumps(compare, indent=2, default=float))
    print(f"[json] {OUT_JSON}")

    # ── Markdown table ──
    lines = []
    lines.append("# Refit compare (old v4 vs new seasonpop)\n")
    lines.append("## π (channel, mean [95% CI])\n")
    lines.append("| channel | old | new | Δmean |")
    lines.append("|---|---|---|---|")
    for ch in CH:
        r = compare["pi"][ch]
        lines.append(f"| {ch} | {_fmt(r['old_mean'], 3)} [{_fmt(r['old_lo'], 3)},{_fmt(r['old_hi'], 3)}]"
                     f" | {_fmt(r['new_mean'], 3)} [{_fmt(r['new_lo'], 3)},{_fmt(r['new_hi'], 3)}]"
                     f" | {_fmt(r['delta'], 4)} |")

    lines.append("\n## R0 (season, mean [95% CI])\n")
    lines.append("| season | old | new | ΔR0% |")
    lines.append("|---|---|---|---|")
    for s in SEAS:
        r = compare["R0"][s]
        lines.append(f"| {s} | {_fmt(r['old_mean'], 3)} [{_fmt(r['old_lo'], 3)},{_fmt(r['old_hi'], 3)}]"
                     f" | {_fmt(r['new_mean'], 3)} [{_fmt(r['new_lo'], 3)},{_fmt(r['new_hi'], 3)}]"
                     f" | {_pct(r['delta_pct'], 2) if r['delta_pct'] is not None else 'n/a'} |")

    lines.append("\n## Averted % (term window, sick_leave & school-closure)\n")
    lines.append("| season | old sick | new sick | zero∈CI old→new | old school | new school |")
    lines.append("|---|---|---|---|---|---|")
    for s in SEAS:
        r = compare["seasons"][s]
        os_ = r["old_sick_term"]; ns_ = r["new_sick_term"]
        oc_ = r["old_school_term"]; nc_ = r["new_school_term"]
        lines.append(f"| {s} | {_fmt(os_['mean'], 3)} [{_fmt(os_['q05'], 3)},{_fmt(os_['q95'], 3)}]"
                     f" | {_fmt(ns_['mean'], 3)} [{_fmt(ns_['q05'], 3)},{_fmt(ns_['q95'], 3)}]"
                     f" | {os_['zero_in_ci']}→{ns_['zero_in_ci']}"
                     f" | {_fmt(oc_['mean'], 3)} [{_fmt(oc_['q05'], 3)},{_fmt(oc_['q95'], 3)}]"
                     f" | {_fmt(nc_['mean'], 3)} [{_fmt(nc_['q05'], 3)},{_fmt(nc_['q95'], 3)}] |")

    lines.append("\n## Δ attack per age (sick, term) — sign flip 강조\n")
    for s in SEAS:
        lines.append(f"\n### {s}")
        lines.append("| age | old (mean/CI, sig) | new (mean/CI, sig) | flip |")
        lines.append("|---|---|---|---|")
        by = compare["seasons"][s]["sick_d_by_age_term"]
        for a in AGES:
            e = by.get(a, {})
            lines.append(f"| {a} | {_pct(e.get('old_mean'), 3)} "
                         f"[{_pct(e.get('old_q05'), 3)},{_pct(e.get('old_q95'), 3)}] "
                         f"sig={e.get('old_sig')} "
                         f"| {_pct(e.get('new_mean'), 3)} "
                         f"[{_pct(e.get('new_q05'), 3)},{_pct(e.get('new_q95'), 3)}] "
                         f"sig={e.get('new_sig')} | {e.get('sign_flip')} |")

    lines.append("\n## 해석 체크\n")
    lines.append("### (a) 병가 시즌별 0∈CI 변화")
    for s in SEAS:
        z = check["sick_zero_in_ci"][s]
        lines.append(f"- {s}: old={z['old']}  new={z['new']}  changed={z['changed']}")
    lines.append("\n### (b) 성인↓/아동↑ 3/3 유지 (sick term)")
    for s in SEAS:
        d = check["adult_down_child_up_3of3"]["by_season"][s]
        lines.append(f"- {s}: adult↓={d['adult_neg']}  child↑={d['child_pos']}  holds={d['holds']}")
    lines.append(f"- all_3: **{check['adult_down_child_up_3of3']['all_hold']}**")
    lines.append("\n### (c) 학교 > 병가 (term)")
    for s in SEAS:
        d = check["school_gt_sick"][s]
        lines.append(f"- {s}: school>{d['sick'] if d['sick'] is None else round(d['sick'], 3)}  "
                     f"→ school>{d['school'] if d['school'] is None else round(d['school'], 3)}  "
                     f"holds={d['school_gt_sick']}")
    lines.append("\n### (d) 시즌 간 아동 재분배 크기 순서 (new)")
    if check.get("seasonal_child_reorder", {}).get("order_new"):
        lines.append(f"- order_new (by |sick_child_d_term|): {check['seasonal_child_reorder']['order_new']}")
    lines.append("")

    lines.append("\n## Flags\n")
    if flags:
        for f in flags: lines.append(f"- {f}")
    else:
        lines.append("- (none)")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines))
    print(f"[md]  {LOG}")


if __name__ == "__main__":
    main()
