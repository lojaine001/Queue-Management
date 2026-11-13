# App image: API, forecast scheduler, push alert checker, operator dashboard.
# One image; docker-compose.yml picks the program with `command`.
# Build context: the repository root.
FROM python:3.10.16-slim-bookworm

# The forecast uses the machine's local clock (naive "now" in Paris time), so the
# container must run in the store's time zone, like the Windows machine did.
ENV TZ=Europe/Paris \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/docker/requirements-app.txt deploy/docker/requirements-app.txt
RUN pip install -r deploy/docker/requirements-app.txt

# Code (venvs, logs, models, .env and other machine files are excluded by .dockerignore)
COPY prediction/ prediction/
COPY run_scheduler.py ./
COPY db/ db/
COPY Queue-Management-System-v2-main/Queue-Management-System-v2-main/ Queue-Management-System-v2-main/Queue-Management-System-v2-main/

# Run as a normal user; the folders that keep data are volumes in docker-compose.yml.
RUN useradd --create-home --uid 1000 iqms \
 && mkdir -p .runtime Queue-Management-System-v2-main/Queue-Management-System-v2-main/models \
             Queue-Management-System-v2-main/Queue-Management-System-v2-main/snapshots logs \
 && chown -R iqms:iqms /app
USER iqms

WORKDIR /app/Queue-Management-System-v2-main/Queue-Management-System-v2-main
EXPOSE 8000 8501
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
