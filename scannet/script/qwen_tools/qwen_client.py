import base64
import hashlib
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from openai import OpenAI
from PIL import Image
import json


def create_qwen_client():
    """Load Qwen credentials from environment variables or a local file."""
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base_url = os.environ.get("QWEN_BASE_URL", "").strip()

    config_path = (
        Path.home() / ".config" / "scannet-sg" / "qwen.json"
    )

    # Environment variables take precedence over the local configuration.
    if not api_key or not base_url:
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)

            if not isinstance(config, dict):
                raise ValueError("Qwen configuration must be a JSON object")

            for name in ("DASHSCOPE_API_KEY", "QWEN_BASE_URL"):
                if not isinstance(config.get(name, ""), str):
                    raise ValueError(f"{name} must be a string")

            if not api_key:
                api_key = config.get("DASHSCOPE_API_KEY", "").strip()

            if not base_url:
                base_url = config.get("QWEN_BASE_URL", "").strip()

    if not api_key or api_key == "在这里填写你的Qwen密钥":
        raise RuntimeError(
            "Set DASHSCOPE_API_KEY in the environment or qwen.json"
        )

    if (
        not base_url
        or base_url == "在这里填写之前成功调用的兼容接口地址"
    ):
        raise RuntimeError(
            "Set QWEN_BASE_URL in the environment or qwen.json"
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
    )


def request_image_tags(
    client,
    image_path,
    model_name,
    prompt,
    prompt_version,
):
    """Send one image and return the full response with provenance."""
    image_path = Path(image_path).expanduser().resolve()
    image_bytes = image_path.read_bytes()

    # Detect the actual format from the same bytes sent to the model.
    with Image.open(BytesIO(image_bytes)) as image:
        image_format = image.format
        width, height = image.size
        image.verify()

    mime_types = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
    }

    if image_format not in mime_types:
        raise ValueError(
            f"Unsupported image format {image_format!r}: {image_path}"
        )

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    mime_type = mime_types[image_format]

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{mime_type};base64,{encoded_image}"
                            ),
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
        max_tokens=2048,
        extra_body={"enable_thinking": False},
    )

    # Preserve the response before downstream parsing or filtering.
    return {
        "provider": "qwen",
        "requested_model": model_name,
        "base_url": str(client.base_url),
        "received_at_utc": datetime.now(timezone.utc).isoformat(),
        "image_path": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "image_size": {
            "width": width,
            "height": height,
        },
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest(),
        "prompt": prompt,
        "request_parameters": {
            "max_tokens": 2048,
            "enable_thinking": False,
        },
        "response": response.model_dump(mode="json"),
    }