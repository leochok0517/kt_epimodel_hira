"""S5+S6: Run existing sens[1..6] with season-pop setup + new posterior.

Monkey-patches:
  sens_common.PI_POST / LOG_R0_POST → merged seasonpop posterior
  sens_common.build_setup(imm=...) → returns dispatch-C (season-aware wrapper)
  sens_common.sim_inc / evaluate_full_stratified / fit_pi_pin → wrappers that
    set DISPATCH season before calling underlying originals
  sens6.sim_inc_n → same wrapper pattern

Output files add _seasonpop suffix.  Existing v4 outputs untouched.

Usage:
  python scripts/sens_seasonpop_runner.py <sens_name>
    where <sens_name> ∈ {sens1,sens2,sens3,sens3ext,sens4,sens5,sens6}
"""
from __future__ import annotations
import os, sys, importlib, subprocess
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

import sens_common as SC
from season_pop_setup import build_seasonwise_setup

MERGED_NPZ = SC.ED / "nuts_seasonpop_merged.npz"


def _load_posterior():
    if not MERGED_NPZ.exists():
        raise FileNotFoundError(f"merged posterior not found: {MERGED_NPZ}")
    d = np.load(MERGED_NPZ)
    return np.asarray(d["pi"]), np.asarray(d["log_R0"])


PI_NEW, LOG_R0_NEW = _load_posterior()
print(f"[posterior] pi={PI_NEW.shape} log_R0={LOG_R0_NEW.shape}")


class SeasonDispatchC:
    """Dispatching C — __getitem__ 은 현재 season 의 값을 반환.
    fit/eval 호출 전 set_season(s) 로 스위치.  imm override 는 build_dispatch(imm=...) 로 재빌드."""
    def __init__(self, imm=None):
        self._imm = imm
        self._C_all = build_seasonwise_setup(imm=imm, gamma_15=SC.GAMMA_15,
                                              use_season_pop=True)
        self._C_by_s = {}
        for s in SC.SEASONS:
            self._C_by_s[s] = dict(
                shared=self._C_all["shared_by_s"][s],
                ngm3=self._C_all["ngm3_by_s"][s],
                st={s: self._C_all["st_by_s"][s]},
                obs=self._C_all["obs"], w=self._C_all["w"],
                nw=self._C_all["nw"], H=self._C_all["H"],
                pop6=self._C_all["pop6_by_s"][s],
                full_obs=self._C_all["full_obs"],
            )
        self._current = SC.SEASONS[0]

    def set_season(self, s):
        assert s in self._C_by_s, f"unknown season {s}"
        self._current = s

    def __getitem__(self, k):
        return self._C_by_s[self._current][k]

    def __contains__(self, k):
        return k in self._C_by_s[self._current]


_CACHE = {}
def _build_dispatch(imm=None):
    """Cache DISPATCH by imm signature to avoid re-building the same setup."""
    key = None if imm is None else tuple(np.asarray(imm).tolist())
    if key not in _CACHE:
        _CACHE[key] = SeasonDispatchC(imm=imm)
    return _CACHE[key]


def _wrap_seasonaware(orig_fn):
    """Return wrapper that sets DISPATCH season before calling orig_fn.
    Accepts C as first positional OR keyword (sens_common 은 sim_inc(**common) 로도 부름)."""
    def wrap(*args, **kw):
        if args:
            C = args[0]; s = args[1] if len(args) > 1 else kw.get("s")
        else:
            C = kw.get("C"); s = kw.get("s")
        if isinstance(C, SeasonDispatchC) and s is not None:
            C.set_season(s)
        return orig_fn(*args, **kw)
    return wrap


def install_patches(mod):
    """Patch sens_common + mod-level bindings."""
    # 1) Replace PI/log_R0 posteriors at both sens_common and module level
    SC.PI_POST = PI_NEW; SC.LOG_R0_POST = LOG_R0_NEW
    if hasattr(mod, "PI_POST"): mod.PI_POST = PI_NEW
    if hasattr(mod, "LOG_R0_POST"): mod.LOG_R0_POST = LOG_R0_NEW

    # 2) Patch build_setup to return dispatch object
    def new_build_setup(imm=None):
        return _build_dispatch(imm=imm)
    SC.build_setup = new_build_setup
    if hasattr(mod, "build_setup"): mod.build_setup = new_build_setup

    # 3) Wrap sim_inc / evaluate_full_stratified / fit_pi_pin
    orig_sim = SC.sim_inc
    orig_eval = SC.evaluate_full_stratified
    orig_fit = SC.fit_pi_pin
    wrap_sim = _wrap_seasonaware(orig_sim)
    wrap_eval = _wrap_seasonaware(orig_eval)
    wrap_fit = _wrap_seasonaware(orig_fit)
    SC.sim_inc = wrap_sim; SC.evaluate_full_stratified = wrap_eval; SC.fit_pi_pin = wrap_fit
    if hasattr(mod, "sim_inc"): mod.sim_inc = wrap_sim
    if hasattr(mod, "evaluate_full_stratified"): mod.evaluate_full_stratified = wrap_eval
    if hasattr(mod, "fit_pi_pin"): mod.fit_pi_pin = wrap_fit

    # 4) sens6-specific: sim_inc_n / evaluate_n
    if hasattr(mod, "sim_inc_n"):
        _orig_n = mod.sim_inc_n
        def wrap_n(C, s, R0, pi, n_stages, w_val, **kw):
            if isinstance(C, SeasonDispatchC): C.set_season(s)
            return _orig_n(C, s, R0, pi, n_stages, w_val, **kw)
        mod.sim_inc_n = wrap_n

    # 5) Redirect output paths (add _seasonpop suffix)
    if hasattr(mod, "NAME"):
        new_name = mod.NAME.replace("_v4", "") + "_seasonpop"
        # ensure suffix uniqueness (no double)
        if not new_name.endswith("_seasonpop"): new_name += "_seasonpop"
        mod.NAME = new_name
        mod.PARTIAL = SC.ED / f"{new_name}.partial.jsonl"
        mod.FINAL = SC.ED / f"{new_name}.json"


SENS_MAP = {
    "sens1": "sens1_piwork_kappa",
    "sens2": "sens2_kappa_upper",
    "sens3": "sens3_R0_transition",
    "sens3ext": "sens3ext_R0_transition",
    "sens4": "sens4_w_sweep",
    "sens5": "sens5_baseline_p",
    "sens6": "sens6_erlang_n",
}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in SENS_MAP:
        print(f"usage: {sys.argv[0]} <sens_name>  options: {list(SENS_MAP)}")
        sys.exit(2)
    name = sys.argv[1]
    mod_name = SENS_MAP[name]
    print(f"[runner] loading module {mod_name}")
    mod = importlib.import_module(mod_name)
    install_patches(mod)
    print(f"[runner] patched.  NAME={mod.NAME}  PARTIAL={mod.PARTIAL.name}")
    try:
        mod.main()
    except Exception as e:
        try:
            subprocess.run(["curl","-s","-d",
                             f"sens_seasonpop {name} FAILED: {type(e).__name__}",
                             "ntfy.sh/hwcho-nuts"], timeout=5, capture_output=True)
        except Exception: pass
        raise


if __name__ == "__main__":
    main()
