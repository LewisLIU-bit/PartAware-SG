# Optional PartAware-SG interfaces

The legacy object graph ABI is unchanged. `object_nodes.nodes` contains object records
with 256-dimensional GroundingDINO visual embeddings and 384-dimensional SBERT text
embeddings. `edge_hypotheses` and `free_space_nodes` keep their existing structure.

## Capture inputs

`run_partaware.py --manifest manifest.json` accepts the existing `scannet_sg_input`
format. RGB/depth/pose paths resolve relative to the manifest. Frame IDs remain
strings. Poses must be metric `T_world_from_camera`; depth must be optical-axis Z,
uint16, and agree with the camera metadata scale.

`--image-dir <scene>` accepts the original ScanNet `_info.txt` and
`frame-XXXXXX.color.jpg`, `.depth.pgm`, `.pose.txt` convention. RGB and depth must
be registered, matching the baseline C++ reader's two-intrinsics projection.

`--processed-scene <scene>` provides `topology_map.json` and `refined_instance/`.
Each selected frame requires `<id>.png` and `<id>_updated_instance.json`.
The mask contains frame-local uint8 labels; the JSON maps those labels to actual
global `instance_id` values. A missing mapping fails rather than guessing ownership.

## Part outputs

`--output` must be fresh and must not replace the baseline scene directory.

| File | Interface |
| --- | --- |
| `partaware_graph.json` | Original graph plus independent `part_nodes`, `part_relations`, schema version, and provenance |
| `frame_parts/<id>.npz` | Boolean `(N,H,W)` masks; overlap is retained independently |
| `frame_parts/<id>.json` | Mask index, detection ID, track ID, label, parent global ID, score, point count |
| `part_<n>.points.npy` | Observed world-coordinate `(N,3)` point cloud in meters |
| `run_config.json` | Explicit arguments, dataset/scene IDs, baseline graph SHA-256 |
| `run_zh.jsonl` | Chinese status messages, evidence counts, and failures |

Part nodes use string IDs such as `part_1`. `parent_id` is a legacy object ID or
null. `status` is `provisional` until the minimum number of distinct frames is
observed; only confirmed, attached parts create `part_of` relations. The independent
`semantic_embedding` uses 1024-dimensional CLIP RN50 masked-crop features. It is
never written into the legacy object's `visual_embedding`.

This implementation adapts OP3DSG knowledge-guided detection and association; it
does not execute OP3DSG's API-based unified semantic reasoning stage. The default
part detector emits its own masks. Optional SAM and geometry cleanups are explicit
CLI switches and should be evaluated separately.

## Object association

Append `--association_mode op3dsg` to either original `openset_ply_map` invocation.
`legacy` remains the default. Existing positional arguments, `--manifest`, and
`--filter_floor` continue to work. The optional mode uses directional neighbor
coverage plus semantic similarity and rejects co-visible track merges. This is an
adaptation for incomplete observations, not an exact OP3DSG reproduction.

## Reproduction

Use the existing `scannet-sg` environment for the saved object frontend. Use
`.venv-vlpart/bin/python` for the new part branch. These environments have distinct
timm versions; do not install VLPart's old full requirements into the baseline
environment. The verified setup uses PyTorch 2.14/CUDA 13 and source-built Detectron2.
Repository revisions and model checksums are recorded in `sources.lock.json` and
`model_checksums.json`.

## Experiment storage

Store outputs under `/home/lewisliu/datasets/scannet-sg-processed/<experiment>_v1`.
The verified `partaware_v1` experiment contains `hypersim/ai_001_002`,
`scannet/scene0000_00`, and external regression/history records. The source tree
contains no generated run results. Only ScanNet and Hypersim dedicated routes
are retained. The original generic manifest ABI is preserved.

## Florence and the common viewer

`get_seg_openset.py --grounding_backend florence|dino` defaults to Florence.
Folder and manifest inputs, masks, frame IDs, object JSON fields, DINO 256D
features, and SBERT 384D embeddings retain the original format. Florence
scores are additional uncalibrated metadata. The shared `JointGrounding`
implementation handles candidate boxes, class NMS, SAM, and duplicate masks;
the portable default uses soft crop evidence, while explicit reference
profiles retain their configured hard negative tests. Neither changes 3D IDs.

`visualize_map_with_nodes` keeps its old arguments and adds optional trailing
arguments: `show_parts=False`, `show_part_points=False`,
`include_provisional_parts=False`, `part_radius=0.035`, `check_only=False`,
and `screenshot_path=None`. The CLI exposes these as flags (`--screenshot`
for the last argument). Child geometry comes from `build_part_overlay`;
the same legacy or GUI renderer displays both layers. Object geometry remains
identical when parts are disabled and remains an unchanged prefix when enabled.
Confirmed parts without a parent are shown only in diagnostic all-track mode.
The compatibility wrapper preserves its existing flag names and infers the
base PLY from the saved part-run configuration when that PLY exists; otherwise
specify `--base-cloud`.
