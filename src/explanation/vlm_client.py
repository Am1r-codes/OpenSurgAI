"""Nemotron VL Vision Language Model client for NVIDIA NIM API.

Provides multimodal analysis of surgical video frames using NVIDIA's
Nemotron Nano VL through the NIM inference microservice.  Mirrors the
NemotronClient pattern from pipeline.py.

The client accepts a base64-encoded JPEG frame and a text prompt,
sending them as a multimodal message to the OpenAI-compatible
chat/completions endpoint.

Usage::

    from src.explanation.vlm_client import VLMClient

    client = VLMClient(api_key="nvapi-...")
    result = client.analyze_frame(
        image_b64="<base64 jpeg>",
        prompt="Describe what you see in this surgical frame.",
    )
    print(result["content"])
"""

from __future__ import annotations

import logging
import os
import re
import time

import httpx

log = logging.getLogger(__name__)

# ── VLM system prompt for surgical frame analysis ─────────────────────

VLM_SYSTEM_PROMPT = """\
You are a surgical vision AI assistant for the OpenSurgAI platform.
You are analyzing a single frame from a laparoscopic cholecystectomy
(gallbladder removal) procedure captured by an endoscopic camera.

YOUR TASK: Provide a clear, educational description of what you observe
in this surgical frame.

You MAY describe:
- Visible anatomical structures (liver, gallbladder, Calot triangle, etc.)
- Surgical instruments present and their positioning
- The apparent surgical phase or action being performed
- Tissue condition, coloration, and any notable findings
- Spatial relationships between structures and instruments

You MUST NOT:
- Give clinical advice or diagnostic opinions
- Claim certainty about conditions you cannot verify from one frame
- Reference pixels, bounding boxes, or detection artifacts
- Use more than 150 words unless asked for a detailed analysis

Be educational, precise, and accessible — like a senior surgeon
teaching during a case review.\
"""

# ── Preset analysis prompts ──────────────────────────────────────────

ANALYSIS_PRESETS: dict[str, str] = {
    "General": (
        "Describe what you see in this surgical frame. Identify visible "
        "anatomy, instruments, and the likely surgical phase."
    ),
    "Safety Check": (
        "Evaluate this frame for the Critical View of Safety (CVS). "
        "Can you identify the cystic duct and cystic artery? Is the "
        "hepatocystic triangle cleared? Rate the CVS achievement."
    ),
    "Instrument ID": (
        "Identify all surgical instruments visible in this frame. "
        "Describe their position and what action they appear to be "
        "performing."
    ),
    "Teaching Point": (
        "What is the most important teaching point a surgical educator "
        "would highlight in this frame? Explain for a trainee."
    ),
}


# ── VLM Client ────────────────────────────────────────────────────────

class VLMClient:
    """Thin wrapper around the NVIDIA Nemotron VL NIM chat completions API.

    Sends multimodal (image + text) requests using the OpenAI-compatible
    format with base64-encoded JPEG frames.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-nano-12b-v2-vl",
        temperature: float = 0.3,
        max_tokens: int | None = 4096,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("NVIDIA_API_KEY")
            or os.environ.get("NEMOTRON_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY "
                "environment variable, or pass api_key=."
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = httpx.Client(timeout=self.timeout)

    def analyze_frame(
        self,
        image_b64: str,
        prompt: str = "Describe this surgical frame.",
        system: str | None = None,
    ) -> dict:
        """Send a frame to Nemotron VL for multimodal analysis.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded JPEG image data (no data URI prefix needed).
        prompt : str
            Text prompt to accompany the image.
        system : str, optional
            System prompt override.  Defaults to VLM_SYSTEM_PROMPT.

        Returns
        -------
        dict with keys: content, model, usage, finish_reason
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Build multimodal user message (OpenAI-compatible format)
        data_uri = f"data:image/jpeg;base64,{image_b64}"
        messages = [
            {
                "role": "system",
                "content": system or VLM_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ]

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens

        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    raw = (
                        choices[0]["message"]["content"].strip()
                        if choices else "(no response)"
                    )
                    # Strip <think> blocks if present
                    content = re.sub(
                        r"<think>.*?</think>", "", raw, flags=re.DOTALL
                    ).strip() or raw

                    return {
                        "content": content,
                        "model": data.get("model", self.model),
                        "usage": data.get("usage", {}),
                        "finish_reason": (
                            choices[0].get("finish_reason", "")
                            if choices else ""
                        ),
                    }
                if resp.status_code == 429 or resp.status_code >= 500:
                    wait = min(2 ** attempt, 30)
                    log.warning(
                        "VLM API %d on attempt %d/%d - retrying in %ds",
                        resp.status_code, attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                # Client error - don't retry
                resp.raise_for_status()
            except httpx.TimeoutException as exc:
                last_exc = exc
                wait = min(2 ** attempt, 30)
                log.warning(
                    "VLM timeout on attempt %d/%d - retrying in %ds",
                    attempt, self.max_retries, wait,
                )
                time.sleep(wait)
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(
                    f"VLM API error {exc.response.status_code}: "
                    f"{exc.response.text}"
                ) from exc

        raise RuntimeError(
            f"VLM API failed after {self.max_retries} retries"
            + (f": {last_exc}" if last_exc else "")
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> VLMClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
