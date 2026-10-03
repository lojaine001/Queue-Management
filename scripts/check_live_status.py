import os
import psutil
import psycopg2
from pathlib import Path
from datetime import datetime
import json

print("=== 1. Active Python Processes in Background ===")
keywords = ["scheduler", "streamlit", "dashboard", "ensemble", "norfair", "owlv2", "detector", "simulator"]
found = 0
for p in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
    try:
        cmd = p.info.get("cmdline") or []
        cmd_str = " ".join(cmd)
        if "python" in p.info["name"].lower() and any(k in cmd_str.lower() for k in keywords):
            uptime = round((datetime.now().timestamp() - p.info["create_time"]) / 60, 1)
            pid = p.info["pid"]
            print(f"PID {pid} (uptime: {uptime} min): {cmd_str[:120]}...")
            found += 1
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

if not found:
    print("No matching background python processes found.")

print("\n=== 2. Scheduler State File Status ===")
sched_path = Path("Queue-Management/.runtime/scheduler.json")
if sched_path.exists():
    try:
        data = json.loads(sched_path.read_text(encoding="utf-8"))
        print("Scheduler status   :", data.get("status"))
        print("Scheduler PID      :", data.get("pid"))
        print("Active Child PID   :", data.get("active_pid"))
        print("Last start (UTC)   :", data.get("last_start"))
        print("Last finish (UTC)  :", data.get("last_finish"))
        print("Heartbeat (UTC)    :", data.get("heartbeat"))
        print("Duration (s)       :", data.get("duration_seconds"))
        print("Exit code          :", data.get("exit_code"))
        print("Last error         :", data.get("last_error"))
    except Exception as e:
        print("Error reading scheduler.json:", e)
else:
    print("Queue-Management/.runtime/scheduler.json does not exist.")

print("\n=== 3. Database Freshness Check ===")
try:
    conn = psycopg2.connect(host="localhost", port=5432, dbname="iqms", user="postgres", password="0000")
    cur = conn.cursor()
    cur.execute("SELECT max(predicted_at) as latest_pred, count(*) as count_last_30m FROM queue_predictions WHERE predicted_at >= NOW() - INTERVAL '30 minutes'")
    row = cur.fetchone()
    print("Latest queue_predictions row  :", row[0])
    print("Rows created in last 30 min   :", row[1])

    cur.execute("SELECT max(timestamp) FROM entrance_events")
    print("Latest entrance_events row    :", cur.fetchone()[0])

    cur.execute("SELECT count(*) FROM queue_predictions WHERE predicted_at >= NOW() - INTERVAL '60 minutes'")
    print("Rows created in last 60 min   :", cur.fetchone()[0])
    cur.close()
    conn.close()
except Exception as e:
    print("Database error:", e)
