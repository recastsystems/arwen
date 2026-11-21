"""MSKF qi probe: logs where qi first goes negative or non-finite.

Active only when MSKF_PROBE_LOG is set. Wraps (after import) the MSKF launch,
the column-cumulus radiation merge, the MSKF radiation feedback merge and the
RRTMGP device validation, and logs minima per call. On the first bad qi it
saves the arrays it holds to MSKF_PROBE_LOG + '.npz'.
"""
import importlib.abc
import os
import sys
import time

LOG = os.environ.get("MSKF_PROBE_LOG")
TARGETS = {"gpuwm.core.mskf", "gpuwm.core.cumulus_feedback", "gpuwm.core.rrtmgp",
           "gpuwm.core.column_cumulus"}
_state = {"mskf": 0, "rad": 0, "saved": False, "last_merge": None, "last_mskf": None}


def _log(msg):
    with open(LOG, "a") as fh:
        fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _stats(cp, a):
    a = cp.asarray(a)
    fin = cp.isfinite(a)
    nbad = int((~fin).sum())
    am = cp.where(fin, a, 0)
    return float(am.min()), float(am.max()), int((am < 0).sum()), nbad


def _fmt(name, s):
    return f"{name}[min={s[0]:.3e} max={s[1]:.3e} neg={s[2]} nonfin={s[3]}]"


def _patch(name, mod):
    import cupy as cp
    if name == "gpuwm.core.mskf":
        orig = mod.launch_mskf

        def launch_mskf(inputs, outputs, **kw):
            st = orig(inputs, outputs, **kw)
            _state["mskf"] += 1
            n = _state["mskf"]
            parts = [f"MSKF call {n}"]
            bad = False
            for key in ("rqicuten", "rqscuten", "rqccuten", "qi_cu", "qc_cu", "cldfra_dp", "cldfra_sh", "nca"):
                if key in outputs:
                    s = _stats(cp, outputs[key])
                    parts.append(_fmt(key, s))
                    if key in ("qi_cu", "cldfra_dp", "cldfra_sh") and (s[2] or s[3]):
                        bad = True
            act = int(cp.asarray(outputs["cu_act"]).sum())
            parts.append(f"active={act}")
            _log(" ".join(parts))
            if bad and not _state.get("saved_mskf"):
                _state["saved_mskf"] = True
                cp.savez(LOG + "-mskf.npz", **{k: cp.asarray(v) for k, v in outputs.items()
                                               if hasattr(v, "shape")},
                         **{"in_" + k: cp.asarray(v) for k, v in inputs.items() if hasattr(v, "shape")})
                _log("saved first bad MSKF output")
            return st
        mod.launch_mskf = launch_mskf
    elif name == "gpuwm.core.cumulus_feedback":
        orig = mod.merge_cumulus_feedback

        def merge_cumulus_feedback(qc, qi, qs, cldfra, feedback, **kw):
            sin = _stats(cp, qi)
            scu = _stats(cp, feedback["qi_cu"])
            cf = cp.asarray(feedback["cldfra_dp"]) + cp.asarray(feedback["cldfra_sh"])
            scf = _stats(cp, cf)
            out = orig(qc, qi, qs, cldfra, feedback, **kw)
            sout = _stats(cp, out[1])
            _log("FBMERGE " + " ".join((_fmt("qi_in", sin), _fmt("qi_cu", scu), _fmt("cf", scf),
                                        _fmt("qi_out", sout))))
            if (sin[2] or sin[3] or sout[2] or sout[3]) and not _state["saved"]:
                _state["saved"] = True
                cp.savez(LOG + "-merge.npz", qi_in=cp.asarray(qi), qi_out=out[1],
                         qc_in=cp.asarray(qc), cldfra=cp.asarray(cldfra),
                         **{"fb_" + k: cp.asarray(v) for k, v in feedback.items()})
                _log("saved first bad merge")
            return out
        mod.merge_cumulus_feedback = merge_cumulus_feedback
    elif name == "gpuwm.core.column_cumulus":
        orig = mod.merge_column_cumulus_clouds

        def merge_column_cumulus_clouds(qc, qi, cldfra, **kw):
            sin = _stats(cp, qi)
            out = orig(qc, qi, cldfra, **kw)
            sout = _stats(cp, out[1])
            _log("COLMERGE " + _fmt("qi_in", sin) + " " + _fmt("qi_out", sout))
            return out
        mod.merge_column_cumulus_clouds = merge_column_cumulus_clouds
    elif name == "gpuwm.core.rrtmgp":
        orig = mod._validate_device_call

        def _validate_device_call(**profiles):
            _state["rad"] += 1
            s = _stats(cp, profiles["qi"])
            _log(f"RADVALID {_state['rad']} " + _fmt("qi", s) + " " + _fmt("qs", _stats(cp, profiles["qs"])))
            return orig(**profiles)
        mod._validate_device_call = _validate_device_call


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in TARGETS:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        orig_exec = loader.exec_module

        def exec_module(module, _orig=orig_exec, _name=name):
            _orig(module)
            try:
                _patch(_name, module)
            except Exception as exc:  # noqa: BLE001
                _log(f"patch failed {_name}: {exc!r}")
        loader.exec_module = exec_module
        return spec


if LOG:
    sys.meta_path.insert(0, _Finder())
    if os.environ.get("MSKF_PROBE_NOLEDGER"):
        # No deferred ledger: model.py falls back to immediate reads when the
        # ledger cannot be built, so every site stops on its own step.
        class _NoLedgerFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name != "gpuwm.core.health_ledger":
                    return None
                for finder in sys.meta_path:
                    if finder is self or finder.__class__ is _Finder or not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(name, path, target)
                    if spec is not None:
                        break
                else:
                    return None
                orig_exec = spec.loader.exec_module

                def exec_module(module, _orig=orig_exec):
                    _orig(module)

                    def _refuse(self, *a, **k):
                        raise RuntimeError("probe: ledger disabled")
                    module.HealthLedger.__init__ = _refuse
                spec.loader.exec_module = exec_module
                return spec
        sys.meta_path.insert(0, _NoLedgerFinder())
