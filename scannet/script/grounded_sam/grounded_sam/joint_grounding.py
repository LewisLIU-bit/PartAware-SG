"""DINO candidates -> optional description evidence -> class NMS -> SAM.

This is an uncalibrated trial. Profiles explicitly define which classes use
Florence. Other classes retain their detector score. No new boxes are generated
by Florence, and SAM quality never changes category confidence.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolve_path(value, parent):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else parent / p).resolve()


def normalized(name):
    return " ".join(name.casefold().split())


def fused_score(detector_score, margin, dino_weight):
    """Log-odds pooling with sigmoid(margin); not a calibrated probability."""
    if not (math.isfinite(detector_score) and math.isfinite(margin)
            and 0 <= detector_score <= 1 and 0 <= dino_weight <= 1):
        raise ValueError("Invalid fusion inputs")
    d = min(1 - 1e-6, max(1e-6, detector_score))
    z = dino_weight * math.log(d / (1-d)) + (1-dino_weight) * margin
    return 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))


def class_nms(records, threshold, score_key):
    """Greedy NMS: suppress overlapping boxes only within the same class."""
    order = sorted(range(len(records)), key=lambda i: (-records[i][score_key], i))
    kept = []
    for i in order:
        a = np.asarray(records[i]["box"], dtype=float)
        duplicate = False
        for j in kept:
            if records[i]["class_id"] != records[j]["class_id"]:
                continue
            b = np.asarray(records[j]["box"], dtype=float)
            wh = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
            intersection = float(np.prod(wh))
            union = float(np.prod(a[2:]-a[:2]) + np.prod(b[2:]-b[:2]) - intersection)
            if union > 0 and intersection / union > threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return kept


def resolve_masks(masks, records, duplicate_iou=0.85, min_pixels=16):
    """Resolve near-identical masks before painting; preserve nested objects.

    Direct, unmodified category detections take precedence over experimental
    description rescues for cross-class duplicates. This is a conservative
    baseline-preservation rule, not a calibrated comparison of the two scores.
    """
    masks = np.asarray(masks, dtype=bool).copy()
    if len(masks) != len(records):
        raise ValueError("Mask/record count mismatch")
    if len(masks) == 0:
        return masks, []
    order = sorted(range(len(records)), key=lambda i: (
        records[i]["description"] is not None, -records[i]["joint_score"], i))
    selected = []
    for i in order:
        if masks[i].sum() < min_pixels:
            continue
        for j in selected:
            intersection = np.count_nonzero(masks[i] & masks[j])
            union = np.count_nonzero(masks[i] | masks[j])
            if union and intersection / union >= duplicate_iou:
                print(f"[MASK_DUPLICATE] drop={i}({records[i]['name']}) "
                      f"keep={j}({records[j]['name']})", flush=True)
                break
        else:
            selected.append(i)
    # Match the main writer's large-first painting, then remove residual slivers.
    labels = np.zeros(masks.shape[1:], dtype=np.int32)
    for i in sorted(selected, key=lambda i: (-int(masks[i].sum()), i)):
        labels[masks[i]] = i + 1
    result = np.zeros_like(masks)
    survivors = []
    for i in selected:
        owned = labels == i + 1
        pixels = int(owned.sum())
        if pixels < min_pixels:
            print(f"[MASK_RESIDUAL] drop={i} pixels={pixels}", flush=True)
            continue
        result[i] = owned
        survivors.append(i)
    return result, survivors


class JointGrounding:
    def __init__(self, config_path):
        self.config_path = Path(config_path).expanduser().resolve()
        self.config = load_json(self.config_path)
        if self.config.get("schema_version") != "joint_grounding_trial_v1":
            raise ValueError("Expected joint_grounding_trial_v1 config")
        self.weight = float(self.config["dino_weight"])
        self.candidate_threshold = float(self.config["candidate_threshold"])
        self.nms_threshold = float(self.config["nms_threshold"])
        self.pre_nms_threshold = float(self.config["pre_nms_threshold"])
        self.max_candidates = int(self.config["max_description_candidates"])
        if (not 0 <= self.weight <= 1 or not 0 < self.candidate_threshold < 1
                or not 0 < self.nms_threshold <= self.pre_nms_threshold <= 1
                or self.max_candidates < 1):
            raise ValueError("Invalid joint grounding parameters")
        self.profiles, self.aliases = {}, {}
        self.proposal_routes, self.profile_queries = {}, {}
        from PIL import Image
        from qwen_tools.manifest_io import load_manifest_images
        for p in self.config["profiles"]:
            canonical = normalized(p["canonical_name"])
            if canonical in self.profiles:
                raise ValueError(f"Duplicate profile: {canonical}")
            score_path = resolve_path(p["score_config"], self.config_path.parent)
            cfg = load_json(score_path)
            if cfg.get("schema_version") != "florence_description_evidence_v1":
                raise ValueError("Unsupported description score config")
            descriptions = cfg["descriptions"]
            if len(descriptions) < 2 or cfg["target_description_id"] not in descriptions:
                raise ValueError("Invalid profile descriptions")
            if any(d not in descriptions or d == cfg["target_description_id"]
                   for d in cfg.get("hard_negative_ids", [])):
                raise ValueError("Invalid hard negative description IDs")
            ref_manifest = resolve_path(p["reference_manifest"], self.config_path.parent)
            dataset, scene, frames = load_manifest_images(ref_manifest)
            if dataset != cfg["dataset"] or scene != cfg["scene_id"]:
                raise ValueError("Reference manifest identity mismatch")
            frame_map = {f["frame_id"]: f for f in frames}
            region_map = {r["region_id"]: r for r in cfg["regions"]}
            references = []
            from qwen_tools.description_scoring import crop_box
            for rid in cfg["reference_region_ids"]:
                r = region_map[rid]
                rgb = Path(frame_map[r["frame_id"]]["image_path"])
                if hashlib.sha256(rgb.read_bytes()).hexdigest() != cfg["image_sha256"][r["frame_id"]]:
                    raise ValueError(f"Reference RGB mismatch: {rid}")
                with Image.open(rgb) as im:
                    references.append(im.convert("RGB").crop(crop_box(r["bbox_xyxy"], *im.size)))
            if not references:
                raise ValueError("No reference images")
            model_dir = resolve_path(cfg["model_dir"], score_path.parent)
            if not (model_dir / "config.json").is_file():
                raise FileNotFoundError(model_dir)
            device = cfg.get("device", "cuda")
            if device not in ("cpu", "cuda"):
                raise ValueError("Invalid Florence device")
            self.profiles[canonical] = {"config": cfg, "references": references,
                                        "model_dir": model_dir, "scorer": None, "baseline": None}
            queries = [canonical] + p.get("aliases", []) + p.get("proposal_terms", [])
            self.profile_queries[canonical] = list(dict.fromkeys(queries))
            for term in queries:
                key = normalized(term)
                if key in self.proposal_routes and self.proposal_routes[key] != canonical:
                    raise ValueError(f"Ambiguous proposal route: {term}")
                self.proposal_routes[key] = canonical
            for alias in [canonical] + p["aliases"]:
                key = normalized(alias)
                if key in self.aliases and self.aliases[key] != canonical:
                    raise ValueError(f"Ambiguous alias: {alias}")
                self.aliases[key] = canonical
        if not self.profiles:
            raise ValueError("At least one description profile is required")
        self.last_details = []

    def canonicalize(self, name):
        return self.aliases.get(normalized(name), name)

    def _release(self, gsam, torch):
        # Stage the three large models instead of keeping all of them on CUDA.
        gsam.grounding_dino_model.model.to("cpu")
        gsam.sam_predictor.reset_image()
        gsam.sam_predictor.model.to("cpu")
        released = set()
        for p in self.profiles.values():
            if p["scorer"] is not None and id(p['scorer']) not in released:
                p["scorer"].model.to("cpu")
                released.add(id(p['scorer']))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _scorer_for_profile(self, profile):
        from qwen_tools.florence_worker import make_florence_scorer
        if profile["scorer"] is None:
            cfg = profile["config"]
            profile["scorer"] = make_florence_scorer(
                profile["model_dir"], cfg.get("device", "cuda"), cfg.get("max_description_tokens", 96))
        return profile["scorer"]

    def log_frame(self, scene_dir, frame_id, instances):
        record = {"message": "完成 Florence 联合评分与实例掩码保存", "frame_id": str(frame_id),
                  "backend": type(self).__name__, "instances": len(instances), "calibrated": False,
                  "objects": [{"name": row["object_name"], "local_id": row["frame_instance_id"],
                               "joint_score": row.get("joint_score"),
                               "description_margin": row.get("description_margin")}
                              for row in instances]}
        with (Path(scene_dir) / "run_zh.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_run_config(self, scene_dir, manifest, threshold, input_jobs=None):
        import inspect
        backend_source = Path(inspect.getfile(type(self)))
        record = {"schema_version": "joint_grounding_trial_v1", "calibrated": False,
                  "joint_config": self.config, "final_threshold": threshold,
                  "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "backend": type(self).__name__,
                  "backend_sha256": hashlib.sha256(backend_source.read_bytes()).hexdigest(),
                  "manifest_sha256": hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if manifest else None,
                  "input_mode": "manifest" if manifest else "legacy_scannet",
                  "input_frames": [{"frame_id": str(fid), "rgb": str(rgb),
                                    "rgb_sha256": hashlib.sha256(Path(rgb).read_bytes()).hexdigest(),
                                    "tags_sha256": hashlib.sha256(Path(tags).read_bytes()).hexdigest()}
                                   for fid, rgb, tags in (input_jobs or [])],
                  "description_profiles": {n: p["config"] for n, p in self.profiles.items()}}
        path = Path(scene_dir) / "run_config.json"
        if path.exists() and load_json(path) != record:
            raise RuntimeError("Run configuration changed; use a fresh output directory")
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    def infer(self, gsam, image, names, threshold, debug_dir=None):
        import cv2
        import torch
        from PIL import Image
        from qwen_tools.description_scoring import crop_box
        self.last_details = []
        if not self.candidate_threshold < threshold < 1:
            raise ValueError("Candidate threshold must be below the final threshold")
        self._release(gsam, torch)
        detector = gsam.grounding_dino_model
        detector.model.to(detector.device)
        height, width = image.shape[:2]
        records, vectors = [], []
        feature_dim = None
        # Single-name queries make class assignment explicit and deterministic.
        for cid, name in enumerate(names):
            canonical = self.proposal_routes.get(normalized(name), normalized(self.canonicalize(name)))
            # Broad proposal terms retrieve boxes; they do not rename input categories.
            aliases = self.profile_queries.get(canonical, [name])
            for query in dict.fromkeys(aliases):
                with torch.inference_mode():
                    detections, features = detector.predict_with_classes(
                        image=image, classes=[query], box_threshold=self.candidate_threshold,
                        text_threshold=0.2)
                features = features.detach().cpu().numpy()
                if features.ndim != 2 or len(features) != len(detections.xyxy):
                    raise RuntimeError("Detector feature/candidate alignment mismatch")
                feature_dim = features.shape[1]
                for box, score, vector in zip(detections.xyxy, detections.confidence, features):
                    score = float(score)
                    box = np.asarray(box, dtype=float)
                    if not np.isfinite(box).all() or not math.isfinite(score) or not 0 < score < 1:
                        continue
                    box[[0,2]] = np.clip(box[[0,2]], 0, width)
                    box[[1,3]] = np.clip(box[[1,3]], 0, height)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    if canonical not in self.profiles and score < threshold:
                        continue
                    records.append({"class_id": cid, "name": name, "canonical": canonical,
                                    "box": box.tolist(), "dino_score": score, "joint_score": score,
                                    "query": query, "feature_index": len(vectors), "description": None})
                    vectors.append(vector.copy())
        # Only near-identical same-class boxes are removed before description scoring.
        records = [records[i] for i in class_nms(records, self.pre_nms_threshold, "dino_score")]
        count = sum(r["canonical"] in self.profiles for r in records)
        if count > self.max_candidates and self.config.get('description_candidate_policy') != 'bounded_all':
            raise RuntimeError(f"{count} description candidates exceed configured limit; none silently truncated")
        self._release(gsam, torch)
        rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        active_scorer = None
        for canonical, profile in self.profiles.items():
            relevant = [r for r in records if r["canonical"] == canonical]
            if not relevant:
                continue
            cfg = profile["config"]
            scorer = self._scorer_for_profile(profile)
            if active_scorer is not scorer:
                if active_scorer is not None:
                    active_scorer.model.to('cpu')
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                scorer.model.to(scorer.device)
                active_scorer = scorer
            descriptions, target = cfg["descriptions"], cfg["target_description_id"]
            if profile["baseline"] is None:
                reference_scores = [scorer.score(im, descriptions) for im in profile["references"]]
                profile["baseline"] = {d: float(np.mean([x[d]["mean_log_likelihood"] for x in reference_scores]))
                                       for d in descriptions}
            batch_size = min(8, max(1, int(self.config.get('description_score_batch', 1))))
            scored = []
            for start in range(0, len(relevant), batch_size):
                crops = [rgb.crop(crop_box(r['box'], width, height)) for r in relevant[start:start+batch_size]]
                batch = (scorer.score_many(crops, descriptions) if batch_size > 1
                         else [scorer.score(crops[0], descriptions)])
                if len(batch) != len(crops):
                    raise RuntimeError('Description crop batch lost its candidate alignment')
                scored.extend(batch)
            for i, (r, values) in enumerate(zip(relevant, scored)):
                ll = {d: v["mean_log_likelihood"] for d,v in values.items()}
                gains = {d: ll[d] - profile["baseline"][d] for d in descriptions}
                other = max((d for d in descriptions if d != target), key=lambda d:gains[d])
                margin = gains[target] - gains[other]
                r["joint_score"] = fused_score(r["dino_score"], margin, self.weight)
                # A speaker reference bank must not normalize away speaker evidence.
                # Compare the target and confusable class on the SAME crop directly.
                hard_negatives = cfg.get("hard_negative_ids", [])
                raw_margins = {d: ll[target] - ll[d] for d in hard_negatives}
                r["description_pass"] = all(v > 0 for v in raw_margins.values())
                decision = "KEEP" if r["description_pass"] and r["joint_score"] >= threshold else "DROP"
                print(f"[DESCRIPTION {i+1}/{len(relevant)}] query={r['query']} "
                      f"DINO={r['dino_score']:.3f} margin={margin:+.3f} "
                      f"raw_competition={raw_margins} joint={r['joint_score']:.3f} {decision}", flush=True)
                r["description"] = {"description_margin": margin, "description_log_likelihood": ll[target],
                                    "description_gain": gains[target], "description_competitor": other,
                                    "scoring_description": descriptions[target], "description_source": cfg.get("description_source", "profile_config"),
                                    "description_support": fused_score(0.5, margin, 0.0),
                                    "description_raw_competition": raw_margins,
                                    "description_pass": r["description_pass"]}
        if active_scorer is not None:
            active_scorer.model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        accepted = [r for r in records if r["joint_score"] >= threshold and r.get("description_pass", True)]
        kept = [accepted[i] for i in class_nms(accepted, self.nms_threshold, "joint_score")]
        if debug_dir is not None:
            folder = Path(debug_dir);folder.mkdir(parents=True,exist_ok=True)
            (folder/"joint_candidates.json").write_text(json.dumps({"candidates":records,"kept":kept},indent=2),encoding="utf-8")
        print(f"[JOINT] candidates={len(records)} passed={len(accepted)} after_nms={len(kept)}", flush=True)
        if not kept:
            return (image.copy(), np.zeros((0,height,width),bool), np.array([],dtype=int),
                    np.array([],dtype=float), np.empty((0,feature_dim or 0),dtype=float))
        self._release(gsam, torch)
        gsam.sam_predictor.model.to("cuda" if torch.cuda.is_available() else "cpu")
        proposals = [{"query_id":f"detection_{i}","name":r["name"],"bbox_xyxy":r["box"]} for i,r in enumerate(kept)]
        masks, sam_details = gsam.segment_external_boxes(image,proposals,outside_penalty=0.5)
        if len(masks) != len(kept) or len(sam_details) != len(kept):
            raise RuntimeError("SAM masks/metadata/candidate alignment mismatch")
        masks, survivors = resolve_masks(
            masks, kept,
            float(self.config.get("mask_duplicate_iou", 0.85)),
            int(self.config.get("min_saved_pixels", 16)),
        )
        print(f"[MASKS] before={len(kept)} final={len(survivors)}", flush=True)
        valid = []
        for i,(r,mask,meta) in enumerate(zip(kept,masks,sam_details)):
            if i not in survivors:
                continue
            detail = {"dino_confidence":r["dino_score"], "joint_score":r["joint_score"],
                      "confidence_type":"uncalibrated_joint_score" if r["description"] else "dino_score",
                      "dino_query":r["query"], "segmentation_box":r["box"],
                      "sam_quality_score":meta.get("sam_quality_score"), "sam_prompt":"box_only"}
            if r["description"]:
                detail.update(r["description"])
            self.last_details.append(detail);valid.append(i)
        return (image.copy(),masks[valid],np.array([kept[i]["class_id"] for i in valid],dtype=int),
                np.array([kept[i]["joint_score"] for i in valid]),
                np.stack([vectors[kept[i]["feature_index"]] for i in valid]) if valid else np.empty((0,feature_dim or 0)))
