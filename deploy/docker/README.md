# IQMS on Docker (new machine: Windows + WSL + NVIDIA GPU)

Everything runs as containers under Docker Engine inside WSL (Ubuntu): no Docker Desktop,
no Windows services, no `start_all.ps1`. Docker is the service manager: every container has
`restart: unless-stopped` and the important ones have health checks.

| Service | What it runs | Port (this machine only) |
|---|---|---|
| `db` | PostgreSQL 17 + TimescaleDB | 127.0.0.1:5432 (`DB_HOST_PORT`) |
| `api` | API (`uvicorn api:app`) | 127.0.0.1:8000 |
| `scheduler` | forecast every 3 min (`run_scheduler.py`) | - |
| `push_alerts` | web push alerts | - |
| `dashboard` | Streamlit operator dashboard | 127.0.0.1:8501 |
| `web` | built web app + `/api` proxy (Caddy) | 127.0.0.1:3001 |
| `ngrok` | public link to `web` (profile `tunnel`) | - |
| `camera_checkout`, `camera_entrance` | camera pipelines on the GPU (profile `cameras`) | - |

## 1. Prepare the machine (once)

On Windows: install the latest **NVIDIA driver** (it provides the GPU to WSL), then in an
administrator PowerShell: `wsl --install -d Ubuntu-24.04` and reboot.

In the Ubuntu terminal:
```bash
# systemd, so Docker starts by itself when WSL starts
printf '[boot]\nsystemd=true\n' | sudo tee /etc/wsl.conf
# (then in PowerShell: wsl --shutdown, and reopen Ubuntu)

# Docker Engine + compose plugin (official repository)
sudo apt-get update && sudo apt-get install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt-get update && sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER && sudo systemctl enable --now docker

# NVIDIA Container Toolkit (lets containers use the GPU)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
```
Close and reopen Ubuntu (for the docker group), then check the GPU from a container:
```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```
It must print the GPU. If it does not, fix the driver / toolkit before going on.

**Starting at Windows boot.** WSL starts when a user opens it. To start it (and so Docker and
every container) at boot without anyone logging in, add a Windows scheduled task, "At startup",
"Run whether user is logged on or not", action: `wsl.exe -d Ubuntu-24.04 --exec /bin/true`.

## 2. Code, settings, models, secrets

Keep the project inside the Linux file system (fast), not under `/mnt/c`:
```bash
git clone https://github.com/arnaudbastide/Queue-Management.git ~/iqms && cd ~/iqms
```
Copy these from the current machine (they are not in git):

| What | From the current machine | To `~/iqms/` |
|---|---|---|
| Settings | `Queue-Management\.env` | `.env` (keep the same values; see below) |
| Checkout detector | `Head-Detector\models\yolov9_s_discrete_headpose_nonms_1x3x480x640.onnx` | `Head-Detector/models/` |
| Entrance detector | `...\Queue-Management-System-v2-main\models\yolov9t.onnx` | `Queue-Management-System-v2-main/Queue-Management-System-v2-main/models/` |
| Trained forecast models (optional, saves the first training) | `...\Queue-Management-System-v2-main\models\versions\` | same folder |
| Push key | the file `VAPID_PRIVATE_KEY_PATH` points to | `secrets/vapid_private_key.pem` |

Add to `.env` (Docker-only settings; the Python services ignore them):
```
COMPOSE_PROFILES=cameras,tunnel        # so `docker compose up -d` starts the cameras and ngrok too
NGROK_AUTHTOKEN=<the ngrok token>      # ngrok config check shows where it is on the old machine
NGROK_DOMAIN=<the reserved domain>.ngrok-free.dev
```
`DB_HOST` and the push key path are set by `docker-compose.yml` and need no change. The camera
logins stay in each `config.yml`. The cameras run on CUDA, so `openvino_device` in `config2.yml`
is ignored.

## 3. Move the data from the current machine

Check the TimescaleDB version on the current machine; the `db` image tag in
`docker-compose.yml` must have the same version (change the tag if needed):
```powershell
& "C:\Program Files\PostgreSQL\17\bin\psql.exe" -U postgres -d iqms -tAc "SELECT extversion FROM pg_extension WHERE extname='timescaledb'"
```
Dump on the current machine (the services can keep running):
```powershell
& "C:\Program Files\PostgreSQL\17\bin\pg_dump.exe" -U postgres -d iqms -Fc -f $HOME\iqms_audit\iqms.dump
```
Copy `iqms.dump` to `~/iqms/` on the new machine, then:
```bash
docker compose up -d db
docker compose cp iqms.dump db:/tmp/iqms.dump
docker compose exec db psql -U postgres -d iqms -c "SELECT timescaledb_pre_restore();"
docker compose exec db pg_restore -U postgres -d iqms --no-owner /tmp/iqms.dump
docker compose exec db psql -U postgres -d iqms -c "SELECT timescaledb_post_restore();"
```
For the switch-over itself, take a last dump right before stopping the old machine, so no
data is lost between the two.

## 4. Start and check

```bash
docker compose up -d --build      # first time: builds the images (about 20 min, about 20 GB of disk)
docker compose ps                 # every service "Up", the checked ones "(healthy)"
curl -s localhost:8000/health     # {"status":"ok", ... "db_pool": ...}
docker compose logs --tail 50 camera_checkout   # "[ORT] Requested provider: cuda", "Snapshot saved"
```
Then open the public link. Without data (a brand-new database) the forecast logs
"No valid data to train" and then "Incomplete model bundle" until a few hours of entrance data
exist; with the moved data it trains on its first run.

## 5. Daily commands

| Task | Command |
|---|---|
| Status and health | `docker compose ps` |
| Logs of one service | `docker compose logs -f --tail 100 api` |
| Restart one service | `docker compose restart scheduler` |
| Stop / start everything | `docker compose stop` / `docker compose up -d` |
| After `git pull` (new code) | `docker compose up -d --build` |
| Change a setting in `.env` | `docker compose up -d` (recreates the services whose settings changed) |
| Do not restart a service after a crash | set `restart: "no"` for it in `docker-compose.yml`, then `docker compose up -d` |
| Database shell | `docker compose exec db psql -U postgres -d iqms` |
| Backup | `docker compose exec db pg_dump -U postgres -d iqms -Fc > iqms_$(date +%F).dump` |

Service logs are kept by Docker (3 files of 20 MB per service); the camera apps also write
their rotating logs to `Head-Detector/LOGs/` and `.../LOGs/`. Data that must survive a
container being recreated lives in the `pgdata` volume (database) and in the mounted folders
(`models/`, `snapshots/`, `.runtime/`).

## Tested so far (laptop, Docker 28, NVIDIA RTX 3060)

All images build; the database, API, scheduler, dashboard and web app run healthy; the API key
and the web proxy behave as on Windows (401 without key, the web app's push calls accepted);
both cameras run on CUDA from a recorded store clip and write to the database; a crashing
service was restarted automatically. Not tested yet: Docker Engine without Desktop, the real
cameras (only reachable from the store network), ngrok, and the data move.
