Set-StrictMode -Version Latest
Set-Location -Path "C:\Users\arnau\Documents\project\rtsp\Retail-Wait-Prediction"

# Activate venv if available
if (Test-Path ".\.venv\Scripts\Activate.ps1") { . .\.venv\Scripts\Activate.ps1 }

Write-Host "--- .runtime listing ---"
if (Test-Path ".runtime") { Get-ChildItem -Path .runtime -Force | Select-Object Name, LastWriteTime, Length | Format-Table -AutoSize } else { Write-Host "NO .runtime DIR" }

Write-Host "\n--- Scheduler processes ---"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'run_scheduler.py' } | Select-Object ProcessId, CommandLine | Format-List

Write-Host "\n--- Streamlit processes ---"
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'streamlit' } | Select-Object ProcessId, CommandLine | Format-List

Write-Host "\n--- DB latest predictions (queue_predictions) ---"
python -c 'import psycopg2; from prediction.pipeline import DB_CONFIG; conn=psycopg2.connect(**DB_CONFIG); cur=conn.cursor(); cur.execute("SELECT max(predicted_at), count(*) FROM queue_predictions"); r=cur.fetchone(); print("max(predicted_at)=", r[0], "count=", r[1]); cur.close(); conn.close()'

Write-Host "\n--- DB sample recent predictions ---"
python -c 'import pandas as pd; from prediction.pipeline import DB_CONFIG; import psycopg2; conn=psycopg2.connect(**DB_CONFIG); df=pd.read_sql("SELECT predicted_at, checkout_arrival, predicted_queue FROM queue_predictions ORDER BY predicted_at DESC LIMIT 5", conn); print(df.to_string(index=False)); conn.close()'

Write-Host "\n--- scheduler.log tail (if exists) ---"
if (Test-Path '.runtime\scheduler.log') { Get-Content '.runtime\scheduler.log' -Tail 200 } else { Write-Host 'NO_SCHEDULER_LOG' }
