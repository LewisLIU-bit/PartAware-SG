import json
from pathlib import Path


def validate_component(value, field_name):
    """Validate a path component without changing its original spelling."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")

    if (
        value in {".", ".."}
        or any(char in value for char in '/\\<>:"|?*\0')
        or any(ord(char) < 32 for char in value)
        or value.endswith((" ", "."))
    ):
        raise ValueError(f"Invalid {field_name}: {value!r}")

    return value


def load_manifest_images(manifest_path):
    """Load RGB records in manifest order while preserving frame IDs."""
    manifest_path = Path(manifest_path).expanduser().resolve()

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if not isinstance(manifest, dict):
        raise ValueError("Manifest must be a JSON object")

    if manifest.get("format") != "scannet_sg_input":
        raise ValueError("Expected a scannet_sg_input manifest")

    dataset = validate_component(manifest.get("dataset"), "dataset")
    scene_id = validate_component(manifest.get("scene_id"), "scene_id")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Manifest must contain a non-empty frames list")

    records = []
    seen_ids = set()

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be a JSON object")

        frame_id = validate_component(frame.get("frame_id"), "frame_id")

        if frame_id in seen_ids:
            raise ValueError(f"Duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)

        rgb = frame.get("rgb")
        if not isinstance(rgb, str) or not rgb.strip():
            raise ValueError(f"Missing RGB path for frame {frame_id}")

        image_path = Path(rgb).expanduser()

        # Resolve relative image paths against the manifest directory.
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path

        image_path = image_path.resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Missing RGB image for frame {frame_id}: {image_path}"
            )

        records.append({
            "frame_id": frame_id,
            "image_path": image_path,
        })

    return dataset, scene_id, records