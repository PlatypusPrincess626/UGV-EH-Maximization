#!/usr/bin/env bash
#
# Paired training sweep: both arms on every seed.
#
#   nohup ./sweep.sh > sweep.out 2>&1 &
#
# Runs sequentially. The environment is ~98% of wall clock and largely
# CPU-bound numpy, so concurrent runs contend for cores rather than
# overlapping usefully -- that is what produced two half-speed
# overlapping runs earlier.
#
# Seeds are the OUTER loop so that complete PAIRS finish early: after
# two runs, seed 1 is done for both arms and can be analysed while the
# rest continue.

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
EPISODES="${EPISODES:-400}"
VARIANTS="${VARIANTS:-lyapunov normal cost}"
# NOTE: was "${CONV_STOP: 0}" -- a space after the colon is not the
# default-value operator, so this expanded to the empty string and
# LTAC_CONV_STOP was never set. Every previous sweep therefore ran
# WITHOUT early stopping regardless of what CONV_STOP was set to.
CONV_STOP="${CONV_STOP:-0}"
POLICIES="${POLICIES:-transformer}"
LOGDIR="${LOGDIR:-sweep_logs}"

# Quadratic-floor sweep for cost_icnn. EPS_VALUES is a third axis
# alongside seeds and variants; it applies only to cost_icnn, whose
# head reads LTAC_EPS_Q. Any other variant ignores it, so listing
# several values while sweeping other arms would run duplicates --
# guarded below.
#
# eps sets the guaranteed lower envelope V >= eps*d^2 and enters the
# reported radius as R = sqrt((rho_eff + L_V D)/eps), so it trades
# certificate tightness against whatever it costs the policy. The
# logged runs cannot settle that trade because all five used
# eps = 1.451; this axis is what settles it.
EPS_VALUES="${EPS_VALUES:-}"

mkdir -p "$LOGDIR"

# One entry per eps value for cost_icnn, one empty entry for every
# other variant, so the loop body is identical either way.
eps_list_for() {
  if [ "$1" = "cost_icnn" ] && [ -n "$EPS_VALUES" ]; then
    echo "$EPS_VALUES"
  else
    echo "__none__"
  fi
}

n_total=0
for s in $SEEDS; do
  for p in $POLICIES; do
    if [ "$p" = "transformer" ]; then
      for v in $VARIANTS; do
        for e in $(eps_list_for "$v"); do
          n_total=$((n_total + 1))
        done
      done
    else
      n_total=$((n_total + 1))
    fi
  done
done

echo "=============================================================="
echo "paired sweep: $n_total runs"
echo "  seeds     : $SEEDS"
echo "  variants  : $VARIANTS"
echo "  policies  : $POLICIES"
echo "  episodes  : $EPISODES"
echo "  eps (icnn): ${EPS_VALUES:-<unset, head default>}"
echo "  conv_stop : $CONV_STOP"
echo "  logs      : $LOGDIR/"
echo "  started   : $(date)"
echo "=============================================================="

n=0
sweep_start=$(date +%s)
for s in $SEEDS; do
  for p in $POLICIES; do
  
    if [ "$p" = "transformer" ]; then
      for v in $VARIANTS; do
       for e in $(eps_list_for "$v"); do
        n=$((n + 1))

        # eps belongs in the run's NAME, not just its environment. Two
        # runs of the same variant and seed at different eps are
        # different experiments, and a shared log path would have the
        # second silently overwrite the first -- and a shared .done
        # marker would skip it entirely.
        if [ "$e" = "__none__" ]; then
          tag="${v}_s${s}"
          eps_env=""
        else
          tag="${v}_eps${e}_s${s}"
          eps_env="$e"
        fi
        log="$LOGDIR/${tag}.out"

        # Skip a run that already completed, so the sweep can be stopped
        # and restarted without repeating work.
        if [ -f "$LOGDIR/${tag}.done" ]; then
          echo "[$n/$n_total] SKIP  ${tag} (already done)"
          continue
        fi

        echo "[$n/$n_total] START ${tag} at $(date +%H:%M:%S) -> $log"
        t0=$(date +%s)

        # LTAC_EPS_Q is exported only when set, so an empty value never
        # shadows the head's own default with the empty string.
        if [ -n "$eps_env" ]; then
          LTAC_EPS_Q="$eps_env" LTAC_CONV_STOP="$CONV_STOP" LTAC_VARIANT="$v" \
            LTAC_SEED="$s" LTAC_EPISODES="$EPISODES" \
            python -u main.py > "$log" 2>&1
        else
          LTAC_CONV_STOP="$CONV_STOP" LTAC_VARIANT="$v" \
            LTAC_SEED="$s" LTAC_EPISODES="$EPISODES" \
            python -u main.py > "$log" 2>&1
        fi
        rc=$?

        t1=$(date +%s)
        mins=$(( (t1 - t0) / 60 ))

        if [ $rc -eq 0 ]; then
          touch "$LOGDIR/${tag}.done"
          echo "[$n/$n_total] DONE  ${tag} in ${mins} min"
        else
          # Keep going. One failed seed should not cost the whole sweep,
          # and the missing .done marker makes it easy to retry later.
          echo "[$n/$n_total] FAIL  ${tag} rc=$rc after ${mins} min -- see $log"
        fi

        elapsed=$(( (t1 - sweep_start) / 60 ))
        remaining=$(( n_total - n ))
        if [ $n -gt 0 ] && [ $remaining -gt 0 ]; then
          eta=$(( elapsed * remaining / n ))
          echo "          elapsed ${elapsed} min, ~${eta} min remaining"
        fi
       done
      done
      
    else 
      n=$((n + 1))
      log="$LOGDIR/${p}_s${s}.out"
      
      # Skip a run that already completed, so the sweep can be stopped
      # and restarted without repeating work.
      if [ -f "$LOGDIR/${p}_s${s}.done" ]; then
        echo "[$n/$n_total] SKIP  ${p} seed ${s} (already done)"
        continue
      fi
      
      echo "[$n/$n_total] START ${p} seed ${s} at $(date +%H:%M:%S) -> $log"
      t0=$(date +%s)
      
      if [ "$p" = "dqn" ]; then
        LTAC_POLICY_TYPE="$p" LTAC_SEED="$s" LTAC_EPISODES="$EPISODES" \
          python -u main.py > "$log" 2>&1
        rc=$?
      else
      	LTAC_POLICY_TYPE="$p" LTAC_SEED="$s" LTAC_EPISODES="30" \
          python -u main.py > "$log" 2>&1
        rc=$?
      fi
      
      t1=$(date +%s)
      mins=$(( (t1 - t0) / 60 ))
      
      if [ $rc -eq 0 ]; then
        touch "$LOGDIR/${p}_s${s}.done"
        echo "[$n/$n_total] DONE  ${p} seed ${s} in ${mins} min"
      else
        # Keep going. One failed seed should not cost the whole sweep,
        # and the missing .done marker makes it easy to retry later.
        echo "[$n/$n_total] FAIL  ${p} seed ${s} rc=$rc after ${mins} min -- see $log"
      fi
      
      elapsed=$(( (t1 - sweep_start) / 60 ))
      remaining=$(( n_total - n ))
      if [ $n -gt 0 ] && [ $remaining -gt 0 ]; then
        eta=$(( elapsed * remaining / n ))
        echo "          elapsed ${elapsed} min, ~${eta} min remaining"
      fi
    fi
  done
done

echo "=============================================================="
echo "sweep finished at $(date), total $(( ($(date +%s) - sweep_start) / 60 )) min"
ls -d rl_csv_*_s*/ 2>/dev/null | sed 's/^/  /'
echo "=============================================================="
