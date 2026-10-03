import json
import sys
pid = 16828
out = {}
try:
    import psutil
    p = psutil.Process(pid)
    out['exe'] = p.exe()
    out['cmdline'] = p.cmdline()
    out['cwd'] = p.cwd()
    try:
        env = p.environ()
        out['env_keys_sample'] = list(env.keys())[:50]
    except Exception as e:
        out['env_err'] = str(e)
except Exception as e:
    out['error'] = str(e)

out_path = 'c:/Users/arnau/Documents/project/rtsp/Retail-Wait-Prediction/scripts/inspect_streamlit_proc.out.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print('WROTE', out_path)
