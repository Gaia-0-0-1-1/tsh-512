#!/bin/bash
# off-box reachability logger — timestamps every Spark outage precisely
# (survives the Spark's death by definition; the one witness the freeze cannot kill)
LOG=C:/Users/Owen/tsh-512/tools/monitor/spark_reachability.log
STATE="unknown"
while true; do
  if ssh -o ConnectTimeout=6 -o BatchMode=yes spark "true" > /dev/null 2>&1; then
    CUR="up"
  else
    CUR="down"
  fi
  if [ "$CUR" != "$STATE" ]; then
    echo "$(date '+%F %T') $STATE -> $CUR" >> "$LOG"
    STATE="$CUR"
  fi
  sleep 30
done
