"""
Modal deployment for Sherlock Investigates conversation inference.

Serves a vLLM OpenAI-compatible endpoint for the fine-tuned R1-distill models.
The existing orchestrator (scripts/conversation/orchestrator.py) requires no changes —
just point AgentConfig.endpoint at the Modal URL printed after deployment.

--- Setup ---
1. pip install modal
2. modal token new                        # authenticate once
3. modal deploy scripts/inference/modal_app.py
4. Copy the printed URL into run_pilot.py / AgentConfig.endpoint
5. Set thinking_mode=True in AgentConfig for R1-distill models

--- Model and adapter selection ---
Set environment variables in a Modal secret named "sherlock-hf":
    HF_TOKEN    = your HuggingFace token (needed for private adapter repos)
    MODEL_ID    = HuggingFace model repo (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
    ADAPTER_ID  = HuggingFace LoRA adapter repo (e.g. <hf-username>/sherlock-r1distill-14b)

--- GPU selection ---
Edit GPU below to match your model size:
    "A10G"      → 7B and 14B models (4-bit, ~12-16GB used of 24GB)
    "A100-40GB" → 32B model (4-bit, ~20-24GB used of 40GB)

--- Cost ---
Modal bills per GPU-second. $30 free credits for new accounts.
Conversations run ~5s/turn × 24 turns = ~2min/conversation.
Estimate: ~200+ conversations per $1 of compute at A10G rates.
Run `modal app stop sherlock-vllm` when not in use to avoid idle charges.

--- Inference from orchestrator ---
After `modal deploy`, the URL looks like:
    https://<your-username>--sherlock-vllm-serve.modal.run

In run_pilot.py:
    AgentConfig(
        model_id="sherlock-adapter",   # matches the lora_modules name below
        endpoint="https://<your-username>--sherlock-vllm-serve.modal.run/v1",
        api_key="modal",               # vLLM accepts any non-empty key
        thinking_mode=True,
    )
"""

import os
import subprocess
import sys
import time

import modal

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GPU = "A10G"           # "A10G" for 7B/14B; "A100-40GB" for 32B
VLLM_PORT = 8000
MINUTES = 60

MODEL_ID  = os.environ.get("MODEL_ID",  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "")   # empty = base model only

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.6",
        "huggingface_hub[cli]",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Persistent volume caches model weights between container starts.
# First cold start downloads from HuggingFace; subsequent starts reuse the cache.
model_volume = modal.Volume.from_name("sherlock-model-cache", create_if_missing=True)
MODEL_CACHE = "/model-cache"

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = modal.App("sherlock-vllm")


@app.cls(
    gpu=GPU,
    image=vllm_image,
    volumes={MODEL_CACHE: model_volume},
    container_idle_timeout=5 * MINUTES,   # spin down after 5min of inactivity
    timeout=30 * MINUTES,
    secrets=[modal.Secret.from_name("sherlock-hf", required=False)],
)
class VLLMServer:
    """Wraps vLLM's OpenAI-compatible server as a Modal class.

    The vLLM process is launched as a subprocess in @enter, then proxied via
    @web_endpoint. This keeps the vLLM server warm for the duration of the
    container's idle timeout window, avoiding per-request cold starts during
    a conversation run.
    """

    @modal.enter()
    def start_server(self) -> None:
        """Download weights and start the vLLM OpenAI server."""
        from huggingface_hub import snapshot_download

        hf_token = os.environ.get("HF_TOKEN")
        model_id  = os.environ.get("MODEL_ID",   MODEL_ID)
        adapter_id = os.environ.get("ADAPTER_ID", ADAPTER_ID)

        # Download base model
        model_cache_dir = f"{MODEL_CACHE}/{model_id.replace('/', '--')}"
        print(f"Downloading base model: {model_id}")
        snapshot_download(
            model_id,
            local_dir=model_cache_dir,
            token=hf_token,
            ignore_patterns=["*.pt", "*.bin"],  # prefer safetensors
        )

        # Download LoRA adapter if specified
        adapter_args: list[str] = []
        if adapter_id:
            adapter_cache_dir = f"{MODEL_CACHE}/{adapter_id.replace('/', '--')}"
            print(f"Downloading adapter: {adapter_id}")
            snapshot_download(
                adapter_id,
                local_dir=adapter_cache_dir,
                token=hf_token,
            )
            # Expose the adapter under the name "sherlock-adapter" so the
            # orchestrator can reference it as model_id="sherlock-adapter"
            adapter_args = [
                "--enable-lora",
                "--lora-modules", f"sherlock-adapter={adapter_cache_dir}",
            ]

        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_cache_dir,
            "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--tensor-parallel-size", "1",
            # guided JSON (schema enforcement) support
            "--guided-decoding-backend", "outlines",
        ] + adapter_args

        self.proc = subprocess.Popen(cmd)
        self._wait_for_ready()

    def _wait_for_ready(self, timeout_s: int = 300) -> None:
        import httpx
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                httpx.get(f"http://localhost:{VLLM_PORT}/health", timeout=2)
                print("vLLM server ready.")
                return
            except Exception:
                time.sleep(5)
        raise RuntimeError(f"vLLM server did not become ready within {timeout_s}s")

    @modal.web_endpoint(method="POST", docs=True)
    def completions(self, request: dict) -> dict:
        """Proxy to vLLM's /v1/chat/completions — OpenAI-compatible."""
        import httpx
        resp = httpx.post(
            f"http://localhost:{VLLM_PORT}/v1/chat/completions",
            json=request,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()

    @modal.web_endpoint(method="GET")
    def health(self) -> dict:
        """Liveness check — returns {"status": "ok"} when vLLM is up."""
        return {"status": "ok"}

    @modal.exit()
    def stop_server(self) -> None:
        if hasattr(self, "proc"):
            self.proc.terminate()


# ---------------------------------------------------------------------------
# Local entrypoint for quick smoke test
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main() -> None:
    """Smoke test: send one turn through the deployed endpoint."""
    import json
    import httpx

    server = VLLMServer()
    url = server.completions.web_url
    print(f"Endpoint: {url}")

    payload = {
        "model": "sherlock-adapter",
        "messages": [
            {"role": "user", "content": "Hello, how are you today?"},
        ],
        "temperature": 0.7,
        "max_tokens": 256,
    }
    resp = httpx.post(url, json=payload, timeout=60)
    print(json.dumps(resp.json(), indent=2))
