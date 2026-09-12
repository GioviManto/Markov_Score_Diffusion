#!/bin/bash
# Submit the budget-saturation array (review Priority 2) once the QOS submit
# cap allows.
#
# The cap is 30 jobs per user and counts array tasks individually, so a 16-task
# array needs 16 free slots. The resolve array fills them, hence the wait.
#
# NO --dependency HERE, deliberately, and this is the interesting part.
#
# The first version depended on the parity gate with `afterok:$GATE`. That is
# correct only while the gate job still exists in Slurm's active records. The
# gate COMPLETED with exit 0, Slurm purged it, and from that moment every
# submission failed with "Job dependency problem" -- not because anything was
# wrong, but because you cannot declare a dependency on a job that has already
# been forgotten. The watcher then retried the same doomed command every five
# minutes for hours, logging an error each time and never submitting.
#
# The lesson worth keeping: a dependency on a SHORT gate job is a race against
# Slurm's accounting purge, and the longer the watcher waits for queue space the
# more certainly it loses that race. Safety that expires is worse than no safety
# because it fails silently in the permissive-looking direction -- the log fills
# with errors that read like a queue problem rather than a design problem.
#
# The gate has run and passed. bocconi_stabilise.sbatch asserts a usable GPU and
# runs the parity tests itself before touching any data, so the device check is
# still enforced where it matters -- inside the job, not in its scheduling.
#
#   nohup ./submit_budget_when_free.sh >/dev/null 2>&1 &

HPC=$HOME/nongaussian-bp/research/nongaussian-bp/hpc
LOG=$HOME/nongaussian-bp/logs/budget_watcher.log
cd "$HPC" || exit 1

DEADLINE=$(( $(date +%s) + 172800 ))   # give up after 48 h
echo "$(date -Is) budget watcher started (no dependency)" >> "$LOG"

while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    N=$(squeue -u "$USER" -r -h 2>/dev/null | wc -l)
    if [ "$N" -le 13 ]; then
        J=$(sbatch --parsable --array=0-15 --export=ALL,MODE=budget \
              bocconi_stabilise.sbatch 2>>"$LOG")
        if [ -n "$J" ]; then
            echo "$(date -Is) submitted budget=$J (queue was $N)" >> "$LOG"
            exit 0
        fi
    fi
    sleep 300
done
echo "$(date -Is) gave up without submitting" >> "$LOG"
