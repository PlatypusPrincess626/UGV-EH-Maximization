#!/usr/bin/env python
"""
Add architecture metadata to checkpoints written before it was recorded.

    LTAC_VARIANT=cost_lipschitz python stamp_arch.py \\
        rl_csv_cost_lipschitz_s*/checkpoints/probe_converged.pt

WHY THIS EXISTS

A state_dict holds tensors and no structure, so a reader has to build
the network before it can load the weights -- and main.py and the
transformer modules read LTAC_ENCODER_C and the rest at IMPORT time.
Checkpoints written before save_full_checkpoint recorded `arch_env`
therefore have to be told their own shape out of band, and if they are
told wrong, load_state_dict(strict=False) leaves the unmatched layers
randomly initialised and says nothing useful.

The weights themselves are complete. Only the description of their
shape is missing, so nothing needs retraining.

VERIFY, THEN STAMP

The obvious version of this script would write whatever LTAC_* happens
to be set and move on, which would replace an unknown architecture
with a confidently wrong one -- worse than the original problem,
because the next reader would trust it.

So the settings are USED FIRST: the network is constructed from them
and the weights loaded with strict=True. Only if every key matches
exactly is the metadata written. A mismatch aborts with the offending
keys, which also makes this a way to discover the right settings by
trying candidates until one loads.

The original file is copied to <name>.bak before anything is written.
"""

import argparse
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# These must be in the environment before main is imported; the caller
# sets them, and they are what gets verified and then recorded.
ARCH_KEYS = ("LTAC_VARIANT", "LTAC_ENCODER_C", "LTAC_ENCODER_QK_C",
             "LTAC_QK_TEMP", "LTAC_LN_EPS", "LTAC_LN_GAMMA_MAX",
             "LTAC_SPECTRAL_C", "LTAC_POLICY_TYPE")

import torch
import main as M


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_model(device="cpu"):
    """Construct the architecture the current environment describes."""
    variant = M.TRANSFORMER_VARIANT
    if variant == "cost_lipschitz":
        from lipschitz_transformer import LipschitzCostTransformerActorCritic as C
        return C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                 sequence_length=M.SEQUENCE_LENGTH,
                 softplus_beta=M.COST_BETA_INIT,
                 beta_gain_target=M.COST_BETA_GAIN_TARGET,
                 spectral_critic=True).to(device)
    if variant in ("cost", "cost_softplus"):
        from cost_transformer import CostTransformerActorCritic as C
        return C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                 sequence_length=M.SEQUENCE_LENGTH,
                 softplus_beta=M.COST_BETA_INIT,
                 beta_gain_target=M.COST_BETA_GAIN_TARGET,
                 spectral_critic=(variant == "cost")).to(device)
    if variant in ("cost_linear", "cost_plain"):
        from ablation_transformer import AblationTransformerActorCritic as C
        return C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
                 sequence_length=M.SEQUENCE_LENGTH,
                 use_spectral=(variant == "cost_linear")).to(device)
    from transformer import TransformerActorCritic as C
    return C(M.VIEW_DISTANCE, scalar_dim=M.SCALAR_DIM,
             sequence_length=M.SEQUENCE_LENGTH).to(device)


def weights_of(blob):
    """The state_dict, whether the blob is format-2 or a bare dict."""
    if isinstance(blob, dict) and "model" in blob and "format" in blob:
        return blob["model"], True
    if hasattr(blob, "state_dict"):
        return blob.state_dict(), False
    return blob, False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify only; write nothing")
    ap.add_argument("--no-backup", action="store_true")
    args = ap.parse_args()

    env = {k: os.environ[k] for k in ARCH_KEYS if os.environ.get(k)}
    if "LTAC_VARIANT" not in env:
        raise SystemExit("set LTAC_VARIANT (and any LTAC_ENCODER_C, "
                         "LTAC_QK_TEMP, ... the run used) before running.")
    print("settings to verify and record:")
    for k, v in sorted(env.items()):
        print("   %-22s %s" % (k, v))
    print()

    ok = bad = 0
    for path in args.checkpoints:
        if not os.path.isfile(path):
            print("  SKIP   %s (not a file)" % path)
            continue
        blob = _load(path)
        sd, is_full = weights_of(blob)

        model = build_model()
        try:
            # strict=True: every key must match in name AND shape. This
            # is the whole point -- a lenient load would accept the
            # wrong architecture and the stamp would then certify it.
            model.load_state_dict(sd, strict=True)
        except RuntimeError as exc:
            first = str(exc).split("\n")[0][:150]
            print("  FAIL   %s\n         %s" % (path, first))
            bad += 1
            continue

        existing = blob.get("arch_env") if isinstance(blob, dict) else None
        if existing:
            print("  HAVE   %s (already stamped: %s)" % (path, existing))
            ok += 1
            continue

        if args.dry_run:
            print("  OK     %s (verified; not written, --dry-run)" % path)
            ok += 1
            continue

        if not is_full:
            # A bare state_dict has nowhere to put metadata, so wrap it.
            # Marked format 1 rather than 2: the optimizer moments, LR
            # position and variance tracker are genuinely absent and
            # claiming format 2 would imply a resumable checkpoint.
            print("  NOTE   %s is a bare state_dict; wrapping it changes "
                  "the file's\n         structure, so any script doing "
                  "load_state_dict(torch.load(p))\n         directly will "
                  "need to read blob['model'] instead." % path)
            blob = {"format": 1, "model": sd}

        blob["arch_env"] = env
        blob["arch"] = {
            "view_distance": M.VIEW_DISTANCE,
            "scalar_dim": M.SCALAR_DIM,
            "sequence_length": M.SEQUENCE_LENGTH,
            "cost_beta_init": M.COST_BETA_INIT,
            "cost_beta_gain_target": M.COST_BETA_GAIN_TARGET,
        }
        if not args.no_backup and not os.path.exists(path + ".bak"):
            shutil.copy2(path, path + ".bak")

        # Write to a temporary file and rename, rather than writing
        # over the original. os.replace is atomic on POSIX, so an
        # interruption mid-write leaves the original intact instead of
        # truncated. Without this the window between opening the file
        # and finishing the write is one in which a power loss or a
        # Ctrl-C destroys the checkpoint, and the .bak is the only
        # thing standing between that and a lost run.
        tmp = path + ".tmp"
        torch.save(blob, tmp)

        # Reload and compare every tensor before committing. Cheap
        # relative to the cost of discovering later that a checkpoint
        # was silently corrupted.
        check, _ = weights_of(_load(tmp))
        mismatch = [k for k in sd
                    if k not in check or not torch.equal(sd[k], check[k])]
        if mismatch:
            os.remove(tmp)
            print("  FAIL   %s\n         round-trip altered %d tensors "
                  "(%s); original untouched"
                  % (path, len(mismatch), mismatch[:3]))
            bad += 1
            continue

        os.replace(tmp, path)
        print("  WROTE  %s (%d tensors verified identical)" % (path, len(sd)))
        ok += 1

    print()
    print("verified %d, failed %d" % (ok, bad))
    if bad:
        print("A failure means the environment does not describe those "
              "weights. Try the other candidate settings -- the encoder "
              "Lipschitz bound printed in each run's .out header "
              "identifies the encoder configuration.")


if __name__ == "__main__":
    main()
