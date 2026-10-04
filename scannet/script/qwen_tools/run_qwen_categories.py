"""Qwen category descriptions, cached instance localization, and text retrieval.

The local text retrieval is diagnostic, not a calibrated visual confidence.
The failed Florence phrase-to-category adapter and candidate-review experiment
have been removed. Existing successful cached localization remains compatible.
"""
import argparse
import hashlib
import json
from pathlib import Path

from .entity_prompt import (
    ENTITY_PROMPT,
    PROMPT_VERSION,
    INDOOR_ENTITY_PROMPT,
    INDOOR_PROMPT_VERSION,
)
from .manifest_io import load_manifest_images
from .response_parser import parse_response


def read_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_new_json(path, data):
    """Create a new JSON file without overwriting an existing result."""
    with path.open("x", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        file.write("\n")


def save_or_check_json(path, data):
    """Reuse an identical result and reject conflicting existing content."""
    if path.exists():
        if read_json(path) != data:
            raise RuntimeError(
                f"Existing output differs: {path}. "
                "Use a new output root for a changed experiment."
            )
        return

    write_new_json(path, data)


def validate_cached_record(record, expected):
    """Prevent reuse across different inputs or experiment settings."""
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(
                f"Cached response has a different {key}. "
                "Use a new output root for a changed experiment."
            )

def run_description_retrieval(args):
    """Compare name-only retrieval with name-and-description retrieval."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    dataset, scene_id, frames = load_manifest_images(args.manifest)
    scene_dir = (
        Path(args.output_root).expanduser().resolve()
        / dataset
        / scene_id
    )

    records = []
    for frame in frames[:args.limit]:
        frame_id = frame["frame_id"]
        tag_path = (
                scene_dir / "refined_instance" / f"{frame_id}.json"
        )
        payload = read_json(tag_path)
        objects = payload.get("objects")

        if not isinstance(objects, list):
            raise ValueError(f"Invalid objects list: {tag_path}")

        for category_index, obj in enumerate(objects):
            name = obj.get("name")
            description = obj.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Invalid category name: {tag_path}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Invalid description: {tag_path}")

            records.append({
                "frame_id": frame_id,
                "category_index": category_index,
                "candidate_id": f"{frame_id}:category_{category_index}",
                "name": name.strip(),
                "description": description.strip(),
                "description_scope": "frame_category",
            })


    if len({item["frame_id"] for item in records}) < 2:
        raise ValueError("Retrieval requires categories from at least two frames")

    model = SentenceTransformer(
        args.text_model,
        device=args.text_device,
    )
    model.max_seq_length = 512

    names = [item["name"] for item in records]
    texts = [
        f"Category: {item['name']}\n"
        f"Visible description: {item['description']}"
        for item in records
    ]

    query_prompt = (
        "Instruct: Retrieve descriptions of potentially corresponding "
        "physical objects across views, considering category compatibility "
        "and visible appearance. Shared surroundings alone are insufficient.\n"
        "Query: "
    )

    def encode(values, prompt=None):
        options = {
            "batch_size": 2,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if prompt is not None:
            options["prompt"] = prompt
        return model.encode(values, **options)

    # Keep query and document encoding consistent with retrieval usage.
    name_docs = encode(names)
    name_queries = encode(names, query_prompt)
    description_docs = encode(texts)
    description_queries = encode(texts, query_prompt)

    name_scores = name_queries @ name_docs.T
    description_scores = description_queries @ description_docs.T

    if not (
        np.isfinite(name_scores).all()
        and np.isfinite(description_scores).all()
    ):
        raise RuntimeError("Non-finite retrieval scores")

    results = []
    for i, item in enumerate(records):
        candidates = [
            j for j, other in enumerate(records)
            if other["frame_id"] != item["frame_id"]
        ]

        def ranked(scores):
            order = sorted(
                candidates, key=lambda j: -float(scores[i, j])
            )[:3]
            return [
                {
                    "record_index": j,
                    "candidate_id": records[j]["candidate_id"],
                    "frame_id": records[j]["frame_id"],
                    "name": records[j]["name"],
                    "score": float(scores[i, j]),
                }
                for j in order
            ]

        name_ranking = ranked(name_scores)
        description_ranking = ranked(description_scores)

        results.append({
            "query_record_index": i,
            "name_only": name_ranking,
            "name_and_description": description_ranking,
        })

        print(f"\n{item['candidate_id']} | {item['name']}")
        for title, ranking in (
            ("NAME", name_ranking),
            ("DESCRIPTION", description_ranking),
        ):
            print(f"  {title}:")
            for hit in ranking:
                print(
                    f"    {hit['candidate_id']} | {hit['name']}"
                    f" | {hit['score']:.4f}"
                )

    output_dir = scene_dir / "text_retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "description_baseline.json"

    report = {
        "schema_version": 1,
        "model": args.text_model,
        "device": args.text_device,
        "max_seq_length": model.max_seq_length,
        "query_prompt": query_prompt,
        "scope": "category_retrieval_not_instance_assignment",
        "records": records,
        "description_document_embeddings": description_docs.tolist(),
        "results": results,
    }

    # Replace only this diagnostic report; leave mapping outputs untouched.
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"\nSaved: {output_path}")


def parse_instance_proposals(record, width, height):
    """Validate proposals and convert normalized boxes to source pixels."""
    import math
    from .response_parser import decode_model_json

    choices = record["response"]["choices"]
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete response; inspect the saved raw JSON")

    content = choices[0]["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Empty response content")

    payload = decode_model_json(content)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object")
    if payload.get("coordinate_system") != "xyxy_1000":
        raise ValueError("Expected coordinate_system=xyxy_1000")

    objects = payload.get("objects")
    if not isinstance(objects, list):
        raise ValueError("Expected an objects list")

    proposals = []
    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            raise ValueError(f"Object {index} is not a JSON object")

        values = {}
        for key in ("name", "appearance"):
            value = obj.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Object {index}: missing {key}")
            values[key] = " ".join(value.split())

        name = values["name"].lower()
        visibility = obj.get("visibility")
        if visibility not in ("full", "partial", "uncertain"):
            raise ValueError(f"Object {index}: invalid visibility")

        box = obj.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Object {index}: expected four coordinates")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError(f"Object {index}: invalid coordinate value")

        x1, y1, x2, y2 = map(float, box)
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError(f"Object {index}: invalid box {box}")

        proposals.append({
            "query_id": f"object_{index + 1:03d}",
            "name": name,
            "description": values["appearance"],
            "grounding_text": f"{name}, {values['appearance']}",
            "visibility": visibility,
            "bbox_xyxy_1000": [x1, y1, x2, y2],
            "bbox_xyxy": [
                x1 * width / 1000, y1 * height / 1000,
                x2 * width / 1000, y2 * height / 1000,
            ],
            "source": "qwen_instance_proposal",
            "status": "unverified",
        })

    return proposals


def run_instance_localization(args):
    """Generate independent visual proposals without changing segmentation."""
    from PIL import Image, ImageDraw
    from .qwen_client import create_qwen_client, request_image_tags
    from .entity_prompt import INSTANCE_LOCATION_PROMPT, INSTANCE_LOCATION_VERSION

    if args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    dataset, scene_id, frames = load_manifest_images(args.manifest)
    output = (
        Path(args.output_root).expanduser().resolve() / dataset / scene_id
        / "instance_proposals" / INSTANCE_LOCATION_VERSION
    )
    output.mkdir(parents=True, exist_ok=True)

    client = create_qwen_client()
    prompt = INSTANCE_LOCATION_PROMPT

    for frame in frames[:args.limit]:
        frame_id, image_path = frame["frame_id"], frame["image_path"]

        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
        width, height = canvas.size

        expected = {
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "requested_model": args.model,
            "base_url": str(client.base_url),
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "prompt_version": INSTANCE_LOCATION_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "request_parameters": {
                "max_tokens": 2048,
                "enable_thinking": False,
            },
        }

        raw_path = output / f"{frame_id}.raw.json"
        result_path = output / f"{frame_id}.proposals.json"

        if raw_path.exists():
            record = read_json(raw_path)
            validate_cached_record(record, expected)
            print(f"[CACHE] {frame_id}", flush=True)
        else:
            if result_path.exists():
                raise RuntimeError(
                    f"Result exists without raw response: {result_path}"
                )

            print(f"[LOCATE] {frame_id}", flush=True)
            record = request_image_tags(
                client=client,
                image_path=image_path,
                model_name=args.model,
                prompt=prompt,
                prompt_version=INSTANCE_LOCATION_VERSION,
            )
            record.update(
                dataset=dataset,
                scene_id=scene_id,
                frame_id=frame_id,
            )
            write_new_json(raw_path, record)
            validate_cached_record(record, expected)

        proposals = parse_instance_proposals(record, width, height)

        save_or_check_json(result_path, {
            "schema_version": INSTANCE_LOCATION_VERSION,
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "image_sha256": expected["image_sha256"],
            "image_size": {"width": width, "height": height},
            "coordinate_system": "source_pixels_xyxy",
            "objects": proposals,
        })

        draw = ImageDraw.Draw(canvas)
        palette = ["#ff4040", "#00cc66", "#3399ff", "#ffcc00", "#ee55ff"]

        for index, obj in enumerate(proposals):
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            color = palette[index % len(palette)]

            draw.rectangle(
                (
                    min(x1, width - 1), min(y1, height - 1),
                    min(x2, width - 1), min(y2, height - 1),
                ),
                outline=color,
                width=3,
            )

            label = f"{obj['query_id']} {obj['name']}"
            label = label.encode("ascii", "replace").decode("ascii")
            pos = (
                min(x1, max(0, width - len(label) * 7)),
                max(0, y1 - 16),
            )
            draw.rectangle(draw.textbbox(pos, label), fill="black")
            draw.text(pos, label, fill=color)

            print(f"  {label}: {obj['bbox_xyxy']}", flush=True)

        image_output = output / f"{frame_id}.boxes.jpg"
        canvas.save(image_output, quality=95)
        print(
            f"[SAVED] {result_path}\n[BOXES] {image_output}",
            flush=True,
        )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--scene_type",
        choices=["generic", "indoor"],
        default="generic",
        help="Select the entity policy for this experiment.",
    )
    parser.add_argument(
        "--text_retrieval_only",
        action="store_true",
        help="Run local retrieval on existing category descriptions",
    )
    parser.add_argument(
        "--text_model",
        default="Qwen/Qwen3-Embedding-0.6B",
    )
    parser.add_argument(
        "--text_device",
        choices=["cpu", "cuda"],
        default="cpu",
    )

    parser.add_argument(
        "--locate_instances",
        action="store_true",
        help="Locate individual objects with the vision model",
    )

    parser.add_argument(
        "--score_config",
        default=None,
        help="Run offline fixed-region description scoring from a JSON config",
    )
    args = parser.parse_args()

    if args.score_config:
        if args.locate_instances or args.text_retrieval_only or args.model:
            parser.error("--score_config cannot be combined with API or text-retrieval modes")
        from .description_scoring import run_description_scoring
        run_description_scoring(args)
        return

    if args.locate_instances:
        if not args.model:
            parser.error("--locate_instances requires --model")

        if args.text_retrieval_only:
            parser.error("--locate_instances must run as a separate mode")

        run_instance_localization(args)
        return
    if args.text_retrieval_only:
        run_description_retrieval(args)
        return

    if not args.model:
        parser.error("--model is required when requesting image categories")

    if args.scene_type == "indoor":
        prompt = INDOOR_ENTITY_PROMPT
        prompt_version = INDOOR_PROMPT_VERSION
    else:
        prompt = ENTITY_PROMPT
        prompt_version = PROMPT_VERSION

    if args.limit <= 0:
        parser.error("--limit must be greater than zero")

    dataset, scene_id, records = load_manifest_images(args.manifest)
    selected_records = records[:args.limit]

    output_dir = (
        Path(args.output_root).expanduser().resolve()
        / dataset
        / scene_id
    )

    raw_dir = output_dir / "vlm_responses"
    parsed_dir = output_dir / "vlm_parsed"
    tag_dir = output_dir / "refined_instance"

    for folder in (raw_dir, parsed_dir, tag_dir):
        folder.mkdir(parents=True, exist_ok=True)

    from .qwen_client import create_qwen_client, request_image_tags

    client = create_qwen_client()

    prompt_hash = hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()

    print(f"Dataset: {dataset}")
    print(f"Scene: {scene_id}")
    print(f"Model: {args.model}")
    print(f"Selected frames: {len(selected_records)}")
    print(f"Output: {output_dir}")

    for frame in selected_records:
        frame_id = frame["frame_id"]
        image_path = frame["image_path"]

        raw_path = raw_dir / f"{frame_id}.json"
        parsed_path = parsed_dir / f"{frame_id}.json"
        tag_path = tag_dir / f"{frame_id}.json"

        expected = {
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_id": frame_id,
            "provider": "qwen",
            "requested_model": args.model,
            "base_url": str(client.base_url),
            "image_sha256": hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest(),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_hash,
            "request_parameters": {
                "max_tokens": 2048,
                "enable_thinking": False,
            },
        }

        if raw_path.exists():
            record = read_json(raw_path)
            validate_cached_record(record, expected)
            print(f"[CACHE] {frame_id}")
        else:
            # Avoid a paid request when unexplained outputs already exist.
            if parsed_path.exists() or tag_path.exists():
                raise RuntimeError(
                    f"Output exists without a raw response for {frame_id}. "
                    "Use a new output root."
                )

            print(f"[REQUEST] {frame_id}", flush=True)

            record = request_image_tags(
                client=client,
                image_path=image_path,
                model_name=args.model,
                prompt=prompt,
                prompt_version=prompt_version,
            )

            record.update({
                "dataset": dataset,
                "scene_id": scene_id,
                "frame_id": frame_id,
            })

            # Save the complete response before parsing it.
            write_new_json(raw_path, record)
            validate_cached_record(record, expected)

        for parse_attempt in range(3):
            try:
                parsed = parse_response(record)
                break
            except ValueError:
                if parse_attempt == 2:
                    raise
                rejected = raw_dir / 'rejected'
                rejected.mkdir(exist_ok=True)
                digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()[:16]
                raw_path.replace(rejected / f'{frame_id}.{digest}.json')
                print(f'模型响应无效，保留原文并重新识别：{frame_id}', flush=True)
                record = request_image_tags(client=client, image_path=image_path, model_name=args.model,
                                             prompt=prompt, prompt_version=prompt_version)
                record.update(dataset=dataset, scene_id=scene_id, frame_id=frame_id)
                write_new_json(raw_path, record)
                validate_cached_record(record, expected)

        # Keep surface categories outside the instance segmentation input.
        tags = {"objects": parsed["objects"]}

        save_or_check_json(parsed_path, parsed)
        save_or_check_json(tag_path, tags)

        names = [item["name"] for item in tags["objects"]]
        print(f"[TAGS] {frame_id}: {names}")

    print(f"Completed {len(selected_records)} frames.")


if __name__ == "__main__":
    main()
