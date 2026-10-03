import json
import base64
import os
import argparse
from pathlib import Path
from PIL import Image
import hashlib

OBJECT_TAGGING_PROMPT = """
You identify candidate object categories for an object-level 3D scene graph
from a single RGB image.

Select visible physical objects and meaningful physical structures.
Use an object-level interpretation rather than listing every visible
pattern, component, or semantic concept.

Selection policy:
- Include recognizable objects and structures supported by the image.
- Fixed or attached structures can qualify. An entity does not have to
  be movable or detached to be included.
- Walls, fences, buildings, doors, signs, and billboards are eligible.
  Do not reject a structure merely because it occupies the background.
- Prefer the whole object over its ordinary components. For example,
  list a car rather than separately listing its wheels and windows.
- Do not list printed logos, lettering, posters depicted within an
  advertisement, surface textures, shadows, or reflections as separate
  physical objects.
- Distinguish a real object from a picture of that object. A car printed
  on a billboard is not a physical car in the scene.
- A separately mounted physical sign may qualify even when attached
  to another structure.
- For this initial object-level experiment, do not create separate
  nodes for continuous ground regions or painted road markings.
  In particular, do not list individual zebra-crossing stripes.
  Region-level nodes will be considered separately.
- Do not infer objects that are hidden or too ambiguous to identify.

Output policy:
- Return one entry per eligible category, even if multiple instances
  of that category are visible. Detection will locate the instances.
- Use concise, lowercase, singular English names suitable for
  text-guided detection.
- Do not include instance numbers, counts, positions, colons, or
  descriptive attributes in the name.
- Use a specific category only when visible evidence supports it.
- Write a short description grounded in visible appearance.
- If same-category instances differ, describe that variation without
  implying that every instance shares all attributes.
- Do not invent brands, ownership, hidden geometry, or exact dimensions.
- Return only valid JSON with the following structure:
  {"objects": [{"name": "category name", "description": "description"}]}
- Both fields must be non-empty strings.
- If nothing qualifies, return {"objects": []}.
- Do not include Markdown fences or additional fields.
""".strip()

def load_manifest_images(manifest_path):
    """Load image records in manifest order without changing frame IDs."""
    manifest_path = Path(manifest_path).expanduser().resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format") != "scannet_sg_input":
        raise ValueError("Expected a scannet_sg_input manifest")

    dataset = manifest.get("dataset")
    scene_id = manifest.get("scene_id")

    for field_name, value in (("dataset", dataset), ("scene_id", scene_id)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing or invalid {field_name}")

    frames = manifest.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Manifest must contain a non-empty frames list")

    records = []
    seen_ids = set()

    for frame in frames:
        if not isinstance(frame, dict):
            raise ValueError("Each frame must be an object")

        frame_id = frame.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.strip():
            raise ValueError("Each frame_id must be a non-empty string")

        if frame_id in seen_ids:
            raise ValueError(f"Duplicate frame_id: {frame_id}")
        seen_ids.add(frame_id)

        rgb_path = frame.get("rgb")
        if not isinstance(rgb_path, str) or not rgb_path.strip():
            raise ValueError(f"Missing RGB path for frame {frame_id}")

        image_path = Path(rgb_path).expanduser()
        if not image_path.is_absolute():
            image_path = manifest_path.parent / image_path
        image_path = image_path.resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Missing RGB image: {image_path}")

        records.append({
            "frame_id": frame_id,
            "image_path": image_path,
        })

    return dataset, scene_id, records

def build_manifest_requests(manifest_path):
    """Build request IDs and a mapping back to the source frames."""
    dataset, scene_id, records = load_manifest_images(manifest_path)

    image_paths = []
    request_ids = []
    requests = {}

    for record in records:
        frame_id = record["frame_id"]

        # Use source identity instead of machine-specific file paths.
        identity = json.dumps(
            [dataset, scene_id, frame_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        request_id = hashlib.sha256(
            identity.encode("utf-8")
        ).hexdigest()

        if request_id in requests:
            raise ValueError(f"Duplicate request ID for frame {frame_id}")

        image_paths.append(record["image_path"])
        request_ids.append(request_id)

        requests[request_id] = {
            "frame_id": frame_id,
            "image_path": str(record["image_path"]),
        }

    request_map = {
        "schema_version": 1,
        "dataset": dataset,
        "scene_id": scene_id,
        "requests": requests,
    }

    return image_paths, request_ids, request_map

def write_batch_jsonl(image_paths,output_path,model_name="gpt-4o-mini",request_ids=None,):
    image_paths = [Path(path) for path in image_paths]

    if request_ids is None:
        # Preserve the legacy naming convention.
        request_ids = [
            f"{path.parent.name}_{path.name}"
            for path in image_paths
        ]
    else:
        request_ids = list(request_ids)

    if len(request_ids) != len(image_paths):
        raise ValueError("Each image must have exactly one request ID")

    if any(
            not isinstance(request_id, str) or not request_id.strip()
            for request_id in request_ids
    ):
        raise ValueError("Request IDs must be non-empty strings")

    if len(set(request_ids)) != len(request_ids):
        raise ValueError("Duplicate request IDs in the batch")

    with open(output_path, "w", encoding="utf-8") as f:
        for img_path, request_id in zip(image_paths, request_ids):
            # Detect the image format from its contents.
            with Image.open(img_path) as image:
                image_format = image.format

            mime_types = {
                "JPEG": "image/jpeg",
                "PNG": "image/png",
            }

            if image_format not in mime_types:
                raise ValueError(
                    f"Unsupported image format {image_format!r}: {img_path}"
                )

            mime_type = mime_types[image_format]

            with open(img_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode("utf-8")

            request = {
                "custom_id": request_id,# helps track responses
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": OBJECT_TAGGING_PROMPT,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{base64_image}"
                                    },
                                },
                            ],
                        }
                    ],
                }
            }
            f.write(json.dumps(request) + "\n")


def write_batch_jsonl_scene(processed_data_folder, raw_images_folder, scene_id, start_frame, end_frame, output_jsonl_path="batch_inputs.jsonl", output_image_paths_path="request_image_paths.txt", model_name="gpt-4.1-nano"): 
    '''
    Write the batch jsonl file for a scene
    Args:
        processed_data_folder: Note: this folder is used to find the image ids to process. THe images are assumed to be in the refined_instance folder with name frameid.png. Only the name is used.
        raw_images_folder: the folder that contains the raw images
        scene_id: the id of the scene
        start_frame: the start frame id
        end_frame: the end frame
    '''
    # Find scans folder that contains the scene_id
    scans_folder_list = [f for f in os.listdir(processed_data_folder) if scene_id in f]
    if len(scans_folder_list) == 0:
        print(f"Scene {scene_id} not found in {processed_data_folder}")
        exit()
        
    # Get the image ids from the refined_instance folder
    refined_instance_folder = os.path.join(processed_data_folder, scans_folder_list[0], "refined_instance")
    image_files = [f for f in os.listdir(refined_instance_folder) if f.endswith(".png")]
    image_ids = [int(f.split(".")[0]) for f in image_files]
    image_ids.sort()

    # print(f"Image ids: {image_ids}")
    
    # Filter the image ids
    image_ids = [i for i in image_ids if i >= start_frame and i <= end_frame]
    # print(f"Filtered image ids: {image_ids}")

    # Get the image paths from the raw_images folder
    image_paths = [os.path.join(raw_images_folder, scans_folder_list[0], f"frame-{i:06d}.color.jpg") for i in image_ids]

    # print(f"Total number of images: {len(image_paths)}")

    write_batch_jsonl(image_paths, output_jsonl_path, model_name)

    # Write the request image paths to a file
    with open(output_image_paths_path, "w") as f:
        for img_path in image_paths:
            f.write(img_path + "\n")

    # Check if the size of the output file is larger than 198MB
    if os.path.getsize(output_jsonl_path) > 198 * 1024 * 1024:
        print(f"The size of the output file is larger than 198MB")
        raise ValueError("The size of the output file is larger than 198MB")
    
    return output_jsonl_path, output_image_paths_path

def write_manifest_batch(manifest_path,output_jsonl_path,output_request_map_path,model_name,):
    """Write batch requests and their source-frame mapping."""
    image_paths, request_ids, request_map = build_manifest_requests(
        manifest_path
    )

    output_jsonl_path = Path(output_jsonl_path).expanduser().resolve()
    output_request_map_path = (
        Path(output_request_map_path).expanduser().resolve()
    )

    if output_jsonl_path == output_request_map_path:
        raise ValueError("Request and mapping files must have different paths")

    for path in (output_jsonl_path, output_request_map_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")

    for path in (output_jsonl_path, output_request_map_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_batch_jsonl(
        image_paths,
        output_jsonl_path,
        model_name=model_name,
        request_ids=request_ids,
    )

    with output_request_map_path.open("x", encoding="utf-8") as f:
        json.dump(request_map, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Dataset: {request_map['dataset']}")
    print(f"Scene: {request_map['scene_id']}")
    print(f"Requests: {len(request_ids)}")
    print(f"Batch file: {output_jsonl_path}")
    print(f"Request mapping: {output_request_map_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_data_folder", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/processed/scans")
    parser.add_argument("--raw_images_folder", type=str, default="/home/cc/chg_ws/ros_ws/topomap_ws/src/data/test/images/scans")
    parser.add_argument("--scene_id", type=str, default="scene0704_01")
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--end_frame", type=int, default=2000)
    parser.add_argument("--output_jsonl_path", type=str, default="batch_inputs.jsonl")
    parser.add_argument("--output_image_paths_path", type=str, default="request_image_paths.txt")
    parser.add_argument("--model_name", type=str, default="gpt-4.1-nano") #gpt-4o-mini
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Input manifest for dataset-independent image selection",
    )
    parser.add_argument(
        "--output_request_map_path",
        type=str,
        default=None,
        help="Output JSON mapping request IDs to source frames",
    )
    args = parser.parse_args()

    if args.manifest:
        if not args.output_request_map_path:
            parser.error(
                "--output_request_map_path is required with --manifest"
            )

        write_manifest_batch(
            manifest_path=args.manifest,
            output_jsonl_path=args.output_jsonl_path,
            output_request_map_path=args.output_request_map_path,
            model_name=args.model_name,
        )
    else:
        output_jsonl_path, output_image_paths_path = write_batch_jsonl_scene(
            args.processed_data_folder,
            args.raw_images_folder,
            args.scene_id,
            args.start_frame,
            args.end_frame,
            args.output_jsonl_path,
            args.output_image_paths_path,
            args.model_name,
        )

