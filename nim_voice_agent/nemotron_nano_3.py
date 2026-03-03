import os
import subprocess

import modal
import modal.experimental

NIM_HTTP_PORT = 9000
STARTUP_TIMEOUT_SECONDS = 30 * 60

app = modal.App("nim-nemotron-nano-3")

ngc_secret = modal.Secret.from_name("ngc-secret")
nim_cache = modal.Volume.from_name("nim-voice-agent", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvcr.io/nim/nvidia/nemotron-3-nano:1.7.0-variant",
        secret=ngc_secret,
        add_python="3.13",
    )
    .env({
        "NIM_HTTP_API_PORT": str(NIM_HTTP_PORT),
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
    port=NIM_HTTP_PORT, 
    proxy_regions=[REGION],
    exit_grace_period=25, # seconds
    startup_timeout=STARTUP_TIMEOUT_SECONDS,
)
class NemotronNano3():

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


