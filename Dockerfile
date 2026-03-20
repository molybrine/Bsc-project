# Dockerfile
# container definition for running the scuffed experiment
# requires NVIDIA GPU (obviously)
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# install python and stuff
RUN apt-get update && apt-get install -y python3.10 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements first (docker layer caching. im smart sometimes)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# copy everything else
COPY . .

# default: show help (because running experiments by accident is bad)
ENTRYPOINT ["python3"]
CMD ["go.py", "--help"]
