import os
import subprocess

import modal
import modal.experimental

NIM_HTTP_PORT = 9000
NIM_GRPC_PORT = 50051
STARTUP_TIMEOUT_SECONDS = 30 * 60

app = modal.App("nim-riva-asr")

ngc_secret = modal.Secret.from_name("ngc-secret")
nim_cache = modal.Volume.from_name("nim-voice-agent", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvcr.io/nim/nvidia/parakeet-0-6b-ctc-en-us:3.1.0",
        secret=ngc_secret,
        add_python="3.13",
    )
    .env({
        "NIM_HTTP_API_PORT": str(NIM_HTTP_PORT),
        "NIM_GRPC_API_PORT": str(NIM_GRPC_PORT),
        "NIM_TAGS_SELECTOR": "mode=str,vad=silero,diarizer=sortformer",
        "NIM_CACHE_PATH": "/opt/nim/.cache",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    .entrypoint([])
)

REGION = "us-west"

@app.cls(
    image=image,
    gpu="H100",
    memory=16384,
    volumes={"/opt/nim/.cache": nim_cache},
    secrets=[ngc_secret],
    timeout=60 * 60,
    min_containers=1,
    region=REGION,
)
@modal.experimental.http_server(
    port=50051, 
    proxy_regions=[REGION],
    h2_enabled=True,
    exit_grace_period=25, # seconds
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
)
class RivaASR():

    @modal.enter()
    def serve(self):

        possible_scripts = [
            "/opt/nim/start_server.sh",
            "/opt/nim/start-server",
        ]

        for script in possible_scripts:
            if os.path.exists(script):
                subprocess.Popen(["bash", script])
                break


