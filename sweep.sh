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
CONV_STOP="${CONV_STOP: 0}"
POLICIES="${POLICIES:-transformer}"
LOGDIR="${LOGDIR:-sweep_logs}"

mkdir -p "$LOGDIR"

n_total=0
for s in $SEEDS; do
  for p in $POLICIES; do
    if [ "$p" = "transformer" ]; then
      for v in $VARIANTS; do
        n_total=$((n_total + 1))
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
        n=$((n + 1))
        log="$LOGDIR/${v}_s${s}.out"

        # Skip a run that already completed, so the sweep can be stopped
        # and restarted without repeating work.
        if [ -f "$LOGDIR/${v}_s${s}.done" ]; then
          echo "[$n/$n_total] SKIP  ${v} seed ${s} (already done)"
          continue
        fi

        echo "[$n/$n_total] START ${v} seed ${s} at $(date +%H:%M:%S) -> $log"
        t0=$(date +%s)

        LTAC_CONV_STOP="$CONV_STOP" LTAC_VARIANT="$v" LTAC_SEED="$s" LTAC_EPISODES="$EPISODES" \
          python -u main.py > "$log" 2>&1
        rc=$?

        t1=$(date +%s)
        mins=$(( (t1 - t0) / 60 ))

        if [ $rc -eq 0 ]; then
          touch "$LOGDIR/${v}_s${s}.done"
          echo "[$n/$n_total] DONE  ${v} seed ${s} in ${mins} min"
        else
          # Keep going. One failed seed should not cost the whole sweep,
          # and the missing .done marker makes it easy to retry later.
          echo "[$n/$n_total] FAIL  ${v} seed ${s} rc=$rc after ${mins} min -- see $log"
        fi

        elapsed=$(( (t1 - sweep_start) / 60 ))
        remaining=$(( n_total - n ))
        if [ $n -gt 0 ] && [ $remaining -gt 0 ]; then
          eta=$(( elapsed * remaining / n ))
          echo "          elapsed ${elapsed} min, ~${eta} min remaining"
        fi
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
