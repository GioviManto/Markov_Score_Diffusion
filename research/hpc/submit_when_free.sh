#!/bin/bash
# Submit the wavelet image and video jobs as soon as the QOS submit cap allows.
#
# The `normal` QOS caps submissions at 30 per user and counts array tasks
# individually, so with 29 finalem tasks queued nothing else fits. This polls
# and submits the three combined jobs the moment there is room, then exits.
# It sleeps almost all of its life, so it costs the login node nothing.
HPC=$HOME/nongaussian-bp/research/nongaussian-bp/hpc
LOG=$HOME/nongaussian-bp/logs/submit_watcher.log
cd "$HPC" || exit 1
DEADLINE=$(( $(date +%s) + 172800 ))   # give up after 48 h

echo "$(date -Is) watcher started" >> "$LOG"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    N=$(squeue -u "$USER" -r -h 2>/dev/null | wc -l)
    if [ "$N" -le 26 ]; then
        SMOKE=$(sbatch --parsable --partition=short_cpu --time=01:00:00 \
                --export=ALL,MODE=smoke bocconi_wavelet.sbatch 2>>"$LOG")
        if [ -n "$SMOKE" ]; then
            echo "$(date -Is) smoke=$SMOKE (queue was $N)" >> "$LOG"
            # Both heavy jobs gate on the smoke test: if the code is broken on
            # this cluster there is no point burning 35 h of compute to find out.
            I=$(sbatch --parsable --dependency=afterok:$SMOKE --partition=compute \
                --time=12:00:00 --export=ALL,MODE=images bocconi_wavelet.sbatch 2>>"$LOG")
            V=$(sbatch --parsable --dependency=afterok:$SMOKE --partition=compute \
                --time=23:00:00 --export=ALL,MODE=videoall bocconi_wavelet.sbatch 2>>"$LOG")
            echo "$(date -Is) images=$I videoall=$V" >> "$LOG"
            echo "$(date -Is) watcher done" >> "$LOG"
            exit 0
        fi
    fi
    sleep 300
done
echo "$(date -Is) watcher gave up after 48h" >> "$LOG"
