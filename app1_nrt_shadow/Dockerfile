# Deliberately NOT based on the official HA add-on base images (which bundle
# s6-overlay init and are tuned for lightweight add-ons) -- this add-on runs
# one long-lived Python service and has a real scientific dependency stack
# (georinex/numpy/scipy/pandas/xarray), so a plain, well-understood Python
# base image keeps the build simple and debuggable. Supervisor accepts any
# valid Docker image as an add-on as long as it starts and stays running.
FROM python:3.11-slim

LABEL io.hass.type="addon"

WORKDIR /app

# gcc/build-essential: some scientific wheels still need to compile small
# extensions on less common platforms; kept for robustness even though
# amd64 wheels usually exist prebuilt. Removed after pip install to keep
# the image small.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc build-essential && \
    apt-get autoremove -y

COPY app/ /app/

# /data is Supervisor's standard per-addon persistent volume -- state and
# logs survive add-on restarts/updates here, not lost on every redeploy.
VOLUME ["/data"]

CMD ["python", "-u", "main.py"]
