"""Category detection and external-box SAM segmentation.

Cleanup release: removed experimental candidate review, long-description
DINO ablations, and Florence/DINO concatenation. Category descriptions are
retained in saved records. External masks remain non-exclusive diagnostics;
they are not ready for the legacy 3D fusion writer.
"""
import os
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print(path)
sys.path.append(path)

import cv2
import json
import numpy as np
from pathlib import Path
import tqdm
import argparse
import hashlib


def render_saved_instances(image_path, json_dir, frame_id):
    """Visualize the exact instance labels consumed by 3D fusion."""
    json_dir = Path(json_dir)

    image = cv2.imread(str(image_path))
    mask = cv2.imread(
        str(json_dir / f"{frame_id}.png"),
        cv2.IMREAD_UNCHANGED,
    )
    records = json.loads(
        (json_dir / f"{frame_id}_instance.json").read_text(
            encoding="utf-8"
        )
    )

    if image is None or mask is None:
        raise RuntimeError(f"Missing RGB or mask for {frame_id}")
    if mask.ndim != 2 or mask.shape != image.shape[:2]:
        raise ValueError(f"RGB/mask shape mismatch for {frame_id}")
    if not isinstance(records, list):
        raise ValueError(f"Expected an instance list for {frame_id}")

    canvas = image.copy()
    labels = []

    for obj in records:
        local_id = int(obj["frame_instance_id"])
        region = mask == local_id
        ys, xs = np.where(region)

        if len(xs) == 0:
            print(f"[EMPTY_INSTANCE] {frame_id} local_id={local_id}")
            continue

        # Deterministic BGR color based on the frame-local instance ID.
        color = (
            int(60 + (local_id * 67) % 180),
            int(60 + (local_id * 97) % 180),
            int(60 + (local_id * 137) % 180),
        )
        canvas[region] = (
            0.65 * image[region].astype(np.float32)
            + 0.35 * np.asarray(color, dtype=np.float32)
        ).astype(np.uint8)

        contours, _ = cv2.findContours(
            region.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(canvas, contours, -1, color, 1)

        name = obj.get("object_name", "unknown")
        confidence = float(obj.get("confidence", 0.0))
        text = f"id={local_id} {name} score={confidence:.2f}"
        labels.append((int(xs.min()), int(ys.min()), text, color))

    # Draw labels last so that other masks cannot paint over them.
    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    for x, y, text, color in labels:
        (tw, th), baseline = cv2.getTextSize(text, font, 0.45, 1)
        x = max(0, min(x, width - tw - 8))
        y = max(th + 8, min(y, height - baseline - 4))

        cv2.rectangle(
            canvas,
            (x, y - th - 5),
            (min(width - 1, x + tw + 6), y + baseline + 3),
            color,
            -1,
        )
        cv2.putText(
            canvas, text, (x + 3, y),
            font, 0.45, (0, 0, 0), 1, cv2.LINE_AA,
        )

    output_dir = json_dir.parent / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{frame_id}.instances.jpg"

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Failed to write {output_path}")

    print(f"[INSTANCE_VIS] {output_path}")

class InstanceSegmenter:
    def __init__(self, visualize=False, confidence_threshold: float = 0.4,diagnostics=False, joint=None):
        from sentence_transformers import SentenceTransformer
        from grounded_sam.grounded_sam.grounded_sam_simple_demo import GroundedSam

        # Support the installed older SentenceTransformer constructor with local models.
        import inspect
        model_options = {'device': 'cpu' if joint else None}
        if 'local_files_only' in inspect.signature(SentenceTransformer).parameters:
            model_options['local_files_only'] = True
        self.bert_model = SentenceTransformer(
            '/home/lewisliu/.cache/torch/sentence_transformers/sentence-transformers_all-MiniLM-L6-v2',
            **model_options)
        self.joint = joint
        self.gsam = GroundedSam()
        self.visualize = visualize
        self.confidence_threshold = confidence_threshold
        self.diagnostics = diagnostics

    def set_folder_and_json(self, json_dir, image_dir):
        self.json_dir = Path(json_dir)
        self.image_dir = Path(image_dir)

    def get_frame_jobs(self, manifest_path=None):
        jobs = []

        if manifest_path is None:
            # Preserve the legacy ScanNet input convention.
            json_files = [
                path for path in self.json_dir.glob("*.json")
                if path.stem.isdecimal()
            ]
            json_files.sort(key=lambda path: int(path.stem))

            for json_path in json_files:
                frame_id = int(json_path.stem)
                image_path = (
                    self.image_dir / f"frame-{frame_id:06d}.color.jpg"
                )
                jobs.append((frame_id, image_path, json_path))

            return jobs

        manifest_path = Path(manifest_path).expanduser().resolve()
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        if manifest.get("format") != "scannet_sg_input":
            raise ValueError("Expected a ScanNet-SG input manifest")

        frames = manifest.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("Manifest must contain a non-empty frame list")

        seen_ids = set()

        # Follow manifest order and read RGB paths explicitly.
        for frame in frames:
            frame_id = frame["frame_id"]

            if (
                not isinstance(frame_id, str)
                or not frame_id
                or frame_id in {".", ".."}
                or any(char in frame_id for char in '/\\<>:"|?*\0')
                or any(ord(char) < 32 for char in frame_id)
                or frame_id.endswith((" ", "."))
            ):
                raise ValueError(f"Invalid frame ID: {frame_id!r}")

            if frame_id in seen_ids:
                raise ValueError(f"Duplicate frame ID: {frame_id}")
            seen_ids.add(frame_id)

            image_path = manifest_path.parent / frame["rgb"]

            # Candidate tags are produced separately from sensor adaptation.
            json_path = self.json_dir / f"{frame_id}.json"

            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            if not json_path.is_file():
                raise FileNotFoundError(
                    f"Missing candidate tags: {json_path}"
                )

            jobs.append((frame_id, image_path, json_path))

        return jobs

    def save_empty_result(self, frame_index, image_path):
        """Write an explicit empty result using the original frame identifier."""
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        mask_path = self.json_dir / f"{frame_index}.png"
        if not cv2.imwrite(str(mask_path), np.zeros(image.shape[:2], dtype=np.uint8)):
            raise RuntimeError(f"Failed to save empty mask: {mask_path}")
        json_path = self.json_dir / f"{frame_index}_instance.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump([], file)
        if self.joint is not None:
            self.joint.log_frame(self.json_dir.parent, frame_index, [])
        if self.visualize:
            render_saved_instances(
                image_path, self.json_dir, frame_index
            )

    def process(self, manifest_path=None, merge_category_descriptions=False):
        jobs = self.get_frame_jobs(manifest_path)
        print(f"Processing {len(jobs)} frames")
        if self.joint is not None:
            existing = [self.json_dir / f"{fid}_instance.json" for fid, _, _ in jobs
                        if (self.json_dir / f"{fid}_instance.json").exists()
                        or (self.json_dir / f"{fid}.png").exists()]
            if existing:
                raise FileExistsError(f"Existing frame output: {existing[0]}; use a fresh output folder")
            self.joint.write_run_config(self.json_dir.parent, manifest_path, self.confidence_threshold, jobs)

        for frame_index, image_path, json_file in tqdm.tqdm(jobs):

            if not image_path.exists():
                print(f"Image not found for {json_file.name}: {image_path}")
                continue

            # Read object info
            with open(json_file, 'r') as f:
                obj_data = json.load(f)


            name_description_dict = {}
            description_groups = {}
            for obj in obj_data['objects']:
                # Skip objects with empty or None names/descriptions
                if obj.get('name') and obj.get('description'):
                    # Ensure name and description are strings
                    if isinstance(obj['name'], str) and isinstance(obj['description'], str):
                        name = obj['name'].strip()
                        description = obj['description'].strip()
                        if name and description:  # Only add if both are non-empty after stripping
                            # Clean the name to make it more suitable for GroundingDINO
                            # Remove special characters and normalize
                            clean_name = name.replace('/', ' ').replace('-', ' ').replace('(', ' ').replace(')', ' ')
                            clean_name = ' '.join(clean_name.split())  # Remove extra whitespace
                            if self.joint is not None:
                                clean_name = self.joint.canonicalize(clean_name)
                            if clean_name:  # Only add if cleaned name is not empty
                                if merge_category_descriptions:
                                    # Match category names without changing the first display name.
                                    category_key = clean_name.casefold()

                                    if category_key not in description_groups:
                                        description_groups[category_key] = {
                                            "name": clean_name,
                                            "descriptions": [],
                                        }

                                    group = description_groups[category_key]

                                    if description not in group["descriptions"]:
                                        group["descriptions"].append(description)

                                    name_description_dict[group["name"]] = " ".join(
                                        group["descriptions"]
                                    )
                                else:
                                    # Preserve the legacy last-description-wins behavior.
                                    name_description_dict[clean_name] = description
                    else:
                        print(f"Warning: Skipping object with non-string name or description in {json_file.name}")
                        print(f"name type: {type(obj['name'])}, description type: {type(obj['description'])}")
                        continue

            # Apply the same structural-background policy to all input modes.
            excluded_categories = {"wall", "floor", "ceiling"}

            def normalize_category(name):
                return " ".join(name.casefold().split())

            name_description_list = []
            excluded_names = []

            for name, description in name_description_dict.items():
                if normalize_category(name) in excluded_categories:
                    excluded_names.append(name)
                    continue

                name_description_list.append(f"{name}: {description}")

            if excluded_names:
                print(f"Excluded background categories: {excluded_names}")
            if len(name_description_list) == 0:
                print(f"No object categories for {json_file.name}")
                self.save_empty_result(frame_index, image_path)
                continue

            # Debug output to help identify issues
            print(f"Processing {json_file.name}: {len(name_description_list)} objects")
            if len(name_description_list) > 0:
                print(f"Sample object: {name_description_list[0]}")

            # Read image
            rgb_img = cv2.imread(str(image_path))
            if rgb_img is None:
                print(f"Failed to read image: {image_path}")
                continue

            # Run segmentation
            try:
                name_list = [name_description.split(":")[0] for name_description in name_description_list]

                # Additional validation: ensure no empty names
                name_list = [name.strip() for name in name_list if name.strip()]

                if len(name_list) == 0:
                    print(f"No valid category names for {json_file.name}")
                    self.save_empty_result(frame_index, image_path)
                    continue

                if self.visualize:
                    print(f"name_description_list: {name_description_list}")
                    print(f"name_list: {name_list}")
                    print(f"Number of valid names: {len(name_list)}")

                print(f"Processing {json_file.name} with {len(name_list)} names: {name_list}")

                debug_dir = None
                if self.diagnostics:
                    debug_dir = (
                        self.json_dir.parent
                        / "diagnostics"
                        / str(frame_index)
                    )

                detection_details = None
                if self.joint is not None:
                    annotated, masks, class_ids, confidences, features = self.joint.infer(
                        self.gsam, rgb_img, name_list, self.confidence_threshold, debug_dir)
                    detection_details = self.joint.last_details
                else:
                    annotated, masks, class_ids, confidences, features = self.gsam.infer(
                        rgb_img,
                        name_list,
                        box_threshold=0.2,
                        text_threshold=0.2,
                        nms_threshold=0.3,
                        confidence_threshold=self.confidence_threshold,
                        debug_dir=debug_dir,
                    )
                effective_names = [name_list[int(cid)] for cid in class_ids]
            except NotImplementedError:
                print("GroundedSam model is not implemented.")
                return
            except Exception as e:
                if self.joint is not None:
                    raise RuntimeError(f"Joint segmentation failed: {frame_index}") from e
                print(f"Error processing {json_file.name}: {str(e)}")
                print(f"name_list: {name_list}")
                print(f"name_description_list: {name_description_list}")
                continue

            if self.visualize and self.joint is None:
                visualization_dir = self.json_dir.parent / "visualizations"
                visualization_dir.mkdir(parents=True, exist_ok=True)

                visualization_path = (
                        visualization_dir / f"{frame_index}.annotated.jpg"
                )

                if not cv2.imwrite(str(visualization_path), annotated):
                    raise RuntimeError(
                        f"Failed to save visualization: {visualization_path}"
                    )

                print(f"Visualization saved: {visualization_path}")

            # Generate mono8 mask. Assume max instance id is 255 in one frame.
            if len(masks) > 0:
                # Filter valid instances
                # Convert masks to boolean arrays before measuring areas and overlaps.
                mask_arrays = [np.asarray(mask, dtype=bool) for mask in masks]
                image_pixel_num_limit = rgb_img.shape[0] * rgb_img.shape[1] * 0.8

                candidate_indices = []
                for idx, class_id in enumerate(class_ids):
                    class_id = int(class_id)
                    area = int(np.count_nonzero(mask_arrays[idx]))
                    confidence = float(confidences[idx])

                    if not 0 <= class_id < len(name_description_list):
                        continue
                    if not np.isfinite(confidence):
                        continue
                    if not 0 < area < image_pixel_num_limit:
                        continue

                    candidate_indices.append(idx)

                # Prefer the more confident detection when masks nearly duplicate each other.
                candidate_indices.sort(
                    key=lambda idx: (
                        -float(confidences[idx]),
                        idx,
                    )
                )

                duplicate_iou_threshold = 0.85
                valid_indices = []

                for idx in candidate_indices:
                    duplicate_of = None

                    for kept_idx in valid_indices:
                        same_name = (
                                effective_names[idx].casefold()
                                == effective_names[kept_idx].casefold()
                        )
                        if not same_name:
                            continue

                        intersection = np.count_nonzero(
                            mask_arrays[idx] & mask_arrays[kept_idx]
                        )
                        union = np.count_nonzero(
                            mask_arrays[idx] | mask_arrays[kept_idx]
                        )
                        iou = intersection / union if union else 0.0

                        if iou >= duplicate_iou_threshold:
                            duplicate_of = kept_idx
                            break

                    if duplicate_of is None:
                        valid_indices.append(idx)
                    else:
                        print(
                            f"[MASK_DUPLICATE] frame={frame_index} "
                            f"drop_detection={idx} keep_detection={duplicate_of}"
                        )

                # Filtered-out detections are a valid empty frame, not a missing result.
                if not valid_indices:
                    self.save_empty_result(frame_index, image_path)
                    continue

                # Preserve the existing painting order for non-duplicate masks.
                valid_indices.sort(
                    key=lambda idx: np.count_nonzero(mask_arrays[idx]),
                    reverse=True,
                )

                if len(valid_indices) > 255:
                    raise RuntimeError(
                        f"Frame {frame_index} has {len(valid_indices)} instances; "
                        "the uint8 instance mask supports at most 255."
                    )

                # Create a label image: 0 is background, 1..N are instances.
                instance_img = np.zeros(
                    rgb_img.shape[:2],
                    dtype=np.uint8,
                )

                # Paint larger masks first; smaller masks overwrite overlaps.
                for local_id, detection_idx in enumerate(
                    valid_indices, start=1
                ):
                    instance_img[mask_arrays[detection_idx]] = local_id

                mask_out_path = self.json_dir / f"{frame_index}.png"
                if not cv2.imwrite(str(mask_out_path), instance_img):
                    raise RuntimeError(f"Failed to save mask: {mask_out_path}")
                #### Prepare instance info
                name_list = []
                description_list = []
                for idx in valid_indices:
                    class_id = class_ids[idx]
                    name_description = name_description_list[class_id]
                    name, description = name_description.split(":", 1)
                    name_list.append(name.strip())
                    description_list.append(description.strip())

                # Preserve the existing category-embedding interface for 3D fusion.
                bert_embeddings = self.bert_model.encode(name_list)

                # Build image_info only for valid instances
                image_info = []
                for i in range(len(valid_indices)):
                    idx = valid_indices[i]
                    entry = {
                        "instance_id": -1,  # -1 means not allocated yet by matching
                        "frame_instance_id": i + 1,
                        "object_name": name_list[i],
                        "object_description": description_list[i],
                        "confidence": float(confidences[idx]), # Orignal confidence list. Use idx instead if i
                        "feature": features[idx].tolist(),  # Orignal feature list. Use idx instead if i
                        "bert_embedding": bert_embeddings[i].tolist()
                    }
                    if detection_details is not None:
                        entry.update(detection_details[idx])
                    entry["description_scope"] = "frame_category"
                    entry["saved_pixels"] = int(
                        np.count_nonzero(instance_img == i + 1)
                    )

                    # Do not save instances fully overwritten by other masks.
                    if entry["saved_pixels"] == 0:
                        continue

                    image_info.append(entry)

                json_out_path = self.json_dir / f"{frame_index}_instance.json"
                with open(json_out_path, 'w') as f:
                    json.dump(image_info, f, indent=2)
                if self.joint is not None:
                    self.joint.log_frame(self.json_dir.parent, frame_index, image_info)
                if self.visualize:
                    render_saved_instances(
                        image_path, self.json_dir, frame_index
                    )
            else:
                print(f"No masks found for {json_file.name}")

                self.save_empty_result(frame_index, image_path)



def run_external_proposal_sam(args):
    """Run external proposals through SAM and preserve independent masks."""
    from qwen_tools.manifest_io import load_manifest_images
    from grounded_sam.grounded_sam.grounded_sam_simple_demo import GroundedSam

    dataset, scene_id, frames = load_manifest_images(args.manifest)
    proposal_dir = Path(args.external_proposals).expanduser().resolve()
    output = Path(args.external_output).expanduser().resolve()
    penalty = args.external_outside_penalty

    if not np.isfinite(penalty) or penalty < 0:
        raise ValueError("Invalid --external_outside_penalty")

    # Validate image identity and output paths before loading model weights.
    jobs = []
    for frame in frames:
        fid, image_path = frame["frame_id"], frame["image_path"]
        proposal_path = proposal_dir / f"{fid}.proposals.json"
        proposal_bytes = proposal_path.read_bytes()
        payload = json.loads(proposal_bytes)

        for key, expected in (
            ("dataset", dataset),
            ("scene_id", scene_id),
            ("frame_id", fid),
            ("coordinate_system", "source_pixels_xyxy"),
            (
                "image_sha256",
                hashlib.sha256(image_path.read_bytes()).hexdigest(),
            ),
        ):
            if payload.get(key) != expected:
                raise ValueError(f"Proposal {key} mismatch: {proposal_path}")

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read {image_path}")

        h, w = image.shape[:2]
        if payload.get("image_size") != {"width": w, "height": h}:
            raise ValueError(f"Proposal image_size mismatch: {proposal_path}")
        if not isinstance(payload.get("objects"), list):
            raise ValueError(f"Missing objects list: {proposal_path}")

        frame_output = output / fid
        if frame_output.exists():
            raise FileExistsError(
                f"Output already exists: {frame_output}. "
                "Choose a new --external_output for a rerun."
            )

        jobs.append((frame, payload, proposal_bytes))

    gsam = GroundedSam(load_dino=False)

    for frame, payload, proposal_bytes in jobs:
        fid, image_path = frame["frame_id"], frame["image_path"]
        image = cv2.imread(str(image_path))

        proposals = list(payload["objects"])

        masks, details = gsam.segment_external_boxes(
            image,
            proposals,
            outside_penalty=penalty,
        )

        raw_masks = masks.copy()

        frame_output = output / fid
        frame_output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(
            frame_output / "raw_masks.npz",
            masks=raw_masks,
        )
        np.savez_compressed(
            frame_output / "masks.npz",
            masks=masks,
        )

        # Preserve overlap; these are not exclusive fusion instance labels.
        paint = np.zeros(image.shape, dtype=np.float32)
        count = masks.sum(axis=0)
        colors = []

        for i, (mask, meta) in enumerate(zip(masks, details)):
            color = tuple(
                int(60 + ((i + 1) * n) % 180)
                for n in (67, 97, 137)
            )
            colors.append(color)
            meta["overlap_pixels"] = int(np.count_nonzero(mask & (count > 1)))
            if not np.any(mask):
                continue
            paint[mask] += np.asarray(color, dtype=np.float32)

            canvas = image.copy()
            canvas[mask] = (
                0.65 * image[mask] + 0.35 * np.asarray(color)
            ).astype(np.uint8)

            obj = meta["proposal"]
            x1, y1, x2, y2 = np.rint(obj["bbox_xyxy"]).astype(int)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            text = f"{obj['query_id']} {obj['name']} | {meta['status']}"
            cv2.putText(
                canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA,
            )
            cv2.putText(
                canvas, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA,
            )

            if not cv2.imwrite(
                str(frame_output / f"mask_{i:03d}.jpg"), canvas
            ):
                raise RuntimeError("Failed to save mask visualization")


        overview = image.copy()
        active = count > 0
        average_color = paint[active] / count[active, None]
        overview[active] = (
            0.65 * image[active] + 0.35 * average_color
        ).astype(np.uint8)

        for mask, meta, color in zip(masks, details, colors):
            if not np.any(mask):
                continue
            obj = meta["proposal"]
            x1, y1, x2, y2 = np.rint(obj["bbox_xyxy"]).astype(int)
            cv2.rectangle(overview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overview,
                f"{obj['query_id']} {obj['name']}",
                (max(0, x1), max(18, y1)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1, cv2.LINE_AA,
            )

        if not cv2.imwrite(str(frame_output / "overview.jpg"), overview):
            raise RuntimeError("Failed to save overview")

        result = {
            "schema_version": "external_sam_v1",
            "dataset": dataset,
            "scene_id": scene_id,
            "frame_id": fid,
            "image_sha256": payload["image_sha256"],
            "image_size": payload["image_size"],
            "proposal_sha256": hashlib.sha256(proposal_bytes).hexdigest(),
            "outside_penalty": penalty,
            "selection_rule": (
                "sam_quality - outside_penalty * outside_box_fraction"
            ),
            "fusion_ready": False,
            "objects": details,
        }

        with (frame_output / "result.json").open(
            "x", encoding="utf-8"
        ) as file:
            json.dump(
                result, file,
                ensure_ascii=False, indent=2, allow_nan=False,
            )

        print(
            f"[EXTERNAL_SAM] {fid}: "
            f"{len(details)} proposals -> {frame_output}",
            flush=True,
        )


def check_if_result_exists(json_folder):
    json_files = sorted(os.listdir(json_folder))
    instance_json_files = [file for file in json_files if file.endswith("_instance.json")]
    json_files = [file for file in json_files if not file.endswith("_instance.json") and file.endswith(".json")]
    print(f"number of instance json files: {len(instance_json_files)}")
    print(f"number of json files: {len(json_files)}")
    return len(instance_json_files) == len(json_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="/media/cc/My Passport/dataset/scannet/images/scans")
    parser.add_argument("--json_folder", type=str, default="/media/cc/Expansion/scannet/processed/openset_scans")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--confidence_threshold", type=float, default=0.4)
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to a scene input manifest; omit for legacy ScanNet mode",
    )
    parser.add_argument(
        "--merge_category_descriptions",
        action="store_true",
        help="Merge distinct descriptions for repeated category names",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Save detection stages and independent SAM masks",
    )
    parser.add_argument(
        "--render_saved",
        action="store_true",
        help="Render saved masks and instance JSONs without model inference",
    )
    parser.add_argument("--external_proposals", type=str, default=None)
    parser.add_argument("--external_output", type=str, default=None)
    parser.add_argument(
        "--external_outside_penalty",
        type=float,
        default=0.5,
    )

    parser.add_argument("--joint_config", default=None, help="Joint DINO/description scoring configuration")
    parser.add_argument("--grounding_backend", choices=["florence", "dino"], default="florence",
                        help="Default Florence crop evidence; select dino for the original detector")
    parser.add_argument("--florence_model_dir", default=None,
                        help="Local Florence checkpoint; defaults to FLORENCE_MODEL_DIR or ~/models/vision/Florence-2-large-ft")
    args = parser.parse_args()
    joint = None
    if args.joint_config:
        if args.grounding_backend == "dino" or args.external_proposals or args.external_output or args.render_saved or args.skip_existing:
            parser.error("--joint_config requires Florence detection and cannot use external/render/skip modes")
        from grounded_sam.grounded_sam.joint_grounding import JointGrounding
        joint = JointGrounding(args.joint_config)
    elif args.grounding_backend == "florence" and not (args.external_proposals or args.external_output or args.render_saved):
        from grounded_sam.grounded_sam.florence_grounding import FlorenceGrounding
        joint = FlorenceGrounding(args.florence_model_dir)
    if args.external_proposals is not None:
        if not args.manifest or not args.external_output:
            parser.error(
                "External proposals require --manifest and --external_output"
            )

        if (
                args.render_saved
                or args.skip_existing
                or args.merge_category_descriptions
        ):
            parser.error("External proposals must run as a separate mode")

        run_external_proposal_sam(args)
        raise SystemExit(0)

    if args.external_output is not None:
        parser.error("--external_output requires --external_proposals")
    if args.render_saved:
        if not args.manifest:
            parser.error("--render_saved requires --manifest")

        manifest_path = Path(args.manifest).expanduser().resolve()
        json_dir = Path(args.json_folder).expanduser().resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        for frame in manifest["frames"]:
            frame_id = str(frame["frame_id"])
            image_path = Path(frame["rgb"])
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path

            render_saved_instances(image_path, json_dir, frame_id)

        raise SystemExit(0)
    if args.manifest:
        args.manifest = str(Path(args.manifest).expanduser().resolve())
        args.json_folder = str(Path(args.json_folder).expanduser().resolve())

        if not Path(args.manifest).is_file():
            parser.error(f"Manifest does not exist: {args.manifest}")

        if not Path(args.json_folder).is_dir():
            parser.error(
                f"Candidate-tag directory does not exist: {args.json_folder}"
            )

        # The legacy skip check counts files rather than checking manifest frames.
        if args.skip_existing:
            parser.error(
                "--skip_existing is not yet supported in manifest mode"
            )
    segmenter = InstanceSegmenter(
        visualize=args.visualize,
        confidence_threshold=args.confidence_threshold,
        diagnostics=args.diagnostics,
        joint=joint,
    )

    if args.manifest:
        print("**********Processing a scene from manifest**********")

        segmenter.set_folder_and_json(
            args.json_folder,
            str(Path(args.manifest).parent),
        )
        segmenter.process(
            manifest_path=args.manifest,
            merge_category_descriptions=args.merge_category_descriptions,
        )

    elif args.json_folder.endswith("openset_scans"):
        print("**********Processing openset scannet**********")
        subfolders = [f for f in os.listdir(args.json_folder) if os.path.isdir(os.path.join(args.json_folder, f))]
        for subfolder in subfolders:
            json_folder = os.path.join(args.json_folder, subfolder, "refined_instance")
            image_folder = os.path.join(args.image_folder, subfolder)
            if args.skip_existing and check_if_result_exists(json_folder):
                print(f"Skipping {subfolder} because it already exists")
                continue
            print(f"Processing {subfolder}")
            segmenter.set_folder_and_json(json_folder, image_folder)
            segmenter.process(
                merge_category_descriptions=args.merge_category_descriptions,
            )
    else:
        print("**********Processing fixed set scannet**********")
        # We assume the json_folder is the refined_instance folder and only process one scene
        if args.skip_existing and check_if_result_exists(args.json_folder):
            print(f"Skipping {args.json_folder} because it already exists")
            exit()
        print(f"Processing {args.json_folder}")
        segmenter.set_folder_and_json(args.json_folder, args.image_folder)
        segmenter.process(
            merge_category_descriptions=args.merge_category_descriptions,
        )
