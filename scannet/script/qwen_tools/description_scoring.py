"""Offline Florence description evidence on fixed regions, without segmentation.

Scores are uncalibrated evidence, not detection probabilities. Descriptions
are decoder targets, never part of the encoder prompt. No API client is used.
"""
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

SCHEMA = "florence_description_evidence_v1"
TASK = "<CAPTION>"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, payload):
    # Atomic cache writes allow an interrupted run to resume safely.
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def safe_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError(f"Invalid region ID: {value!r}")
    return value


def crop_box(values, width, height):
    if (not isinstance(values, list) or len(values) != 4
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
        raise ValueError(f"Invalid source-pixel box: {values}")
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Box outside {width}x{height}: {values}")
    return [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]


def summarize(region_scores, reference_ids, target_id):
    """Subtract a fixed reference mean, then compare alternative descriptions."""
    descriptions = list(next(iter(region_scores.values())))
    if target_id not in descriptions or len(descriptions) < 2:
        raise ValueError("A target and at least one competitor are required")
    if not reference_ids or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Reference IDs must be nonempty and unique")
    for scores in region_scores.values():
        if set(scores) != set(descriptions) or not all(math.isfinite(x) for x in scores.values()):
            raise ValueError("Inconsistent or non-finite region scores")
    baseline = {d: sum(region_scores[r][d] for r in reference_ids) / len(reference_ids)
                for d in descriptions}
    rows = []
    for region_id, scores in region_scores.items():
        if region_id in reference_ids:
            continue
        gains = {d: scores[d] - baseline[d] for d in descriptions}
        rivals = [d for d in descriptions if d != target_id]
        strongest = max(rivals, key=lambda d: gains[d])
        rows.append({
            "region_id": region_id,
            "target_id": target_id,
            "mean_log_likelihood": scores[target_id],
            "reference_gain": gains[target_id],
            "raw_margin": scores[target_id] - max(scores[d] for d in rivals),
            "corrected_margin": gains[target_id] - gains[strongest],
            "strongest_competitor": strongest,
            "all_description_scores": scores,
            "all_description_gains": gains,
            "calibrated_confidence": None,
        })
    return baseline, rows


class FlorenceDescriptionScorer:
    """Reusable scoring backend; one region and one target sequence per forward."""
    def __init__(self, model_dir, device="cuda", max_tokens=96):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        self.torch = torch
        self.device = device
        self.max_tokens = max_tokens
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select cpu explicitly in the config")
        self.dtype = torch.float16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), trust_remote_code=True, local_files_only=True)
        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
        # Transformers' static import scan treats Florence's conditional FlashAttention import as mandatory.
        # Eager attention never executes that branch; scope the workaround to this model load only.
        from unittest.mock import patch
        from transformers.dynamic_module_utils import get_imports
        def eager_imports(filename):
            imports = get_imports(filename)
            if Path(filename).name == "modeling_florence2.py":
                source = Path(filename).read_text()
                if "if is_flash_attn_2_available():" in source:
                    imports = [module for module in imports if module != "flash_attn"]
            return imports
        with patch("transformers.dynamic_module_utils.get_imports", eager_imports):
            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_dir), config=config, trust_remote_code=True, local_files_only=True,
                torch_dtype=self.dtype, attn_implementation="eager").to(device).eval()

    def score(self, image, descriptions):
        torch = self.torch
        # The encoder sees only the crop and a fixed caption task.
        inputs = self.processor(text=TASK, images=image, return_tensors="pt")
        inputs = {k: v.to(device=self.device,
                          dtype=self.dtype if v.is_floating_point() else v.dtype)
                  for k, v in inputs.items()}
        result = {}
        for description_id, text in descriptions.items():
            encoded = self.processor.tokenizer(
                text, return_tensors="pt", add_special_tokens=True,
                return_special_tokens_mask=True, truncation=False)
            labels = encoded["input_ids"].to(self.device)
            if labels.shape[1] > self.max_tokens:
                raise ValueError(f"Description too long: {description_id}; no silent truncation")
            keep = (encoded["attention_mask"].bool()
                    & ~encoded["special_tokens_mask"].bool()).to(self.device)
            if not keep.any():
                raise ValueError(f"No ordinary tokens: {description_id}")
            # Florence shifts decoder labels internally. Do not shift twice.
            with torch.inference_mode():
                output = self.model(**inputs, labels=labels, use_cache=False, return_dict=True)
                logits = output.logits.float()
                if logits.shape[:2] != labels.shape:
                    raise RuntimeError("Decoder logits and target sequence are misaligned")
                log_probs = torch.log_softmax(logits, dim=-1)
                target_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                values = target_log_probs[keep]
                if not torch.isfinite(values).all():
                    raise RuntimeError(f"Non-finite token scores: {description_id}")
                mean = float(values.mean().item())
                token_ids = labels[keep].cpu().tolist()
                logs = values.cpu().tolist()
            result[description_id] = {
                "text": text,
                "mean_log_likelihood": mean,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "tokens": self.processor.tokenizer.convert_ids_to_tokens(token_ids),
                "token_log_probabilities": logs,
            }
            del output, logits, log_probs, target_log_probs, values
        return result


def run_description_scoring(args):
    """Read a fixed-region experiment and write independent, resumable reports."""
    from PIL import Image, ImageDraw, ImageOps
    from .manifest_io import load_manifest_images
    config_path = Path(args.score_config).expanduser().resolve()
    config = read_json(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ValueError(f"Expected schema_version={SCHEMA}")
    if args.limit <= 0:
        raise ValueError("--limit must be positive")
    descriptions = config.get("descriptions")
    if not isinstance(descriptions, dict) or len(descriptions) < 2:
        raise ValueError("Provide at least two descriptions")
    for key, value in descriptions.items():
        safe_id(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid description: {key}")
    target = config["target_description_id"]
    if target not in descriptions:
        raise ValueError("Unknown target_description_id")
    model_dir = Path(config["model_dir"]).expanduser()
    if not model_dir.is_absolute():
        model_dir = config_path.parent / model_dir
    model_dir = model_dir.resolve()
    if not (model_dir / "config.json").is_file():
        raise FileNotFoundError(f"Local model config missing: {model_dir}")
    device = config.get("device", "cuda")
    max_tokens = config.get("max_description_tokens", 96)
    if device not in ("cpu", "cuda") or type(max_tokens) is not int or max_tokens < 2:
        raise ValueError("Invalid device or max_description_tokens")
    dataset, scene, frames = load_manifest_images(args.manifest)
    if config.get("dataset") != dataset or config.get("scene_id") != scene:
        raise ValueError("Config dataset/scene does not match manifest")
    frame_map = {f["frame_id"]: f for f in frames[:args.limit]}
    regions = config.get("regions")
    if not isinstance(regions, list) or not regions:
        raise ValueError("regions must be a nonempty list")
    prepared, images, seen = [], {}, set()
    for region in regions:
        rid = safe_id(region["region_id"])
        if rid in seen:
            raise ValueError(f"Duplicate region: {rid}")
        seen.add(rid)
        fid = region["frame_id"]
        if fid not in frame_map:
            raise ValueError(f"Frame {fid} not selected; check manifest and --limit")
        if fid not in images:
            path = Path(frame_map[fid]["image_path"])
            if digest(path) != config["image_sha256"].get(fid):
                raise ValueError(f"Image identity mismatch: {fid}")
            with Image.open(path) as im:
                images[fid] = im.convert("RGB")
        im = images[fid]
        prepared.append({**region, "crop_xyxy": crop_box(region["bbox_xyxy"], *im.size)})
    reference_ids = config.get("reference_region_ids")
    if (not isinstance(reference_ids, list) or not reference_ids
            or len(reference_ids) != len(set(reference_ids))
            or not set(reference_ids) < seen):
        raise ValueError("Use unique reference IDs and at least one separate test region")

    output = (Path(args.output_root).expanduser().resolve() / dataset / scene
              / "description_scores" / SCHEMA)
    model_files = [{"name": str(p.relative_to(model_dir)), "size": p.stat().st_size,
                    "mtime_ns": p.stat().st_mtime_ns}
                   for p in sorted(model_dir.rglob("*")) if p.is_file()
                   and p.suffix in (".json", ".py", ".safetensors", ".bin", ".txt", ".model")]
    provenance = {"schema_version": SCHEMA, "config": config,
                  "implementation_sha256": digest(__file__), "model_dir": str(model_dir),
                  "model_files": model_files, "task": TASK, "regions": prepared,
                  "versions": {p: importlib.metadata.version(p)
                               for p in ("torch", "transformers", "Pillow")},
                  "special_tokens_scored": False, "diagnostic_only": True}
    audit_path = output / "experiment.json"
    if audit_path.exists():
        if read_json(audit_path) != provenance:
            raise RuntimeError("Experiment changed; use a new --output_root")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Existing output has no experiment record; use a new output root")
    output.mkdir(parents=True, exist_ok=True)
    if not audit_path.exists():
        save_json(audit_path, provenance)
    cache_dir = output / "region_scores"
    crop_dir = output / "crops"
    cache_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)
    scorer, matrix, cards = None, {}, []
    for index, region in enumerate(prepared):
        rid = region["region_id"]
        crop = images[region["frame_id"]].crop(region["crop_xyxy"])
        crop.save(crop_dir / f"{rid}.png")
        card = Image.new("RGB", (340, 230), "white")
        thumb = ImageOps.contain(crop, (330, 180))
        card.paste(thumb, ((340-thumb.width)//2, 30+(180-thumb.height)//2))
        draw = ImageDraw.Draw(card)
        draw.text((5, 5), rid, fill="black")
        draw.text((5, 212), "REFERENCE" if rid in reference_ids else "TEST (not ground truth)", fill="black")
        cards.append(card)
        cache_path = cache_dir / f"{rid}.json"
        if cache_path.exists():
            record = read_json(cache_path)
            if record.get("region") != region or set(record.get("scores", {})) != set(descriptions):
                raise RuntimeError(f"Invalid cache: {cache_path}")
            print(f"[CACHE] {rid}", flush=True)
        else:
            print(f"[SCORE {index+1}/{len(prepared)}] {rid}", flush=True)
            if scorer is None:
                scorer = FlorenceDescriptionScorer(model_dir, device, max_tokens)
            record = {"region": region, "scores": scorer.score(crop, descriptions)}
            save_json(cache_path, record)
        matrix[rid] = {d: record["scores"][d]["mean_log_likelihood"] for d in descriptions}
    sheet = Image.new("RGB", (680, 230*math.ceil(len(cards)/2)), "#cccccc")
    for i, card in enumerate(cards):
        sheet.paste(card, ((i%2)*340, (i//2)*230))
    sheet.save(output / "regions_overview.jpg", quality=95)
    baseline, rows = summarize(matrix, reference_ids, target)
    report = {"schema_version": SCHEMA, "diagnostic_only": True,
              "calibrated": False, "reference_region_ids": reference_ids,
              "reference_mean_log_likelihood": baseline, "descriptions": descriptions,
              "results": rows}
    save_json(output / "scores.json", report)
    lines = ["region_id\tmean_log_likelihood\treference_gain\traw_margin\tcorrected_margin\tstrongest_competitor"]
    for row in rows:
        line = (f"{row['region_id']}\t{row['mean_log_likelihood']:.6f}\t"
                f"{row['reference_gain']:.6f}\t{row['raw_margin']:.6f}\t"
                f"{row['corrected_margin']:.6f}\t{row['strongest_competitor']}")
        lines.append(line)
        print(line, flush=True)
    (output / "scores.tsv").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"[SAVED] {output}\nUncalibrated evidence only; no segmentation was changed.", flush=True)
