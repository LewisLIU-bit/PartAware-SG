# PartAware-SG interfaces

The main Hypersim construction path applies the existing multi-frame floor
policy automatically for declared meter-scale Z-up data. The Python component's `floor_filter.json`
records the accepted height, 1 cm removal band and background/frame support.
Cleared points do not receive object identities; the background PLY retains them.
The legacy fallback uses its existing floor filter, with the same guards,
and records the accepted height and support in its step log.

Evaluation JSON adds `object_count_consistency`, containing `predicted`,
`annotated`, `ratio`, `absolute_log_ratio`, `log_base`, `status` and
`absolute_count_error`. Ratios use observed independent GT objects, excluding
walls, floors, ceilings and parts. With positive counts the error is
`abs(log(predicted / annotated))`; lower is better. Zero predictions have null
error and status `infinite_no_predictions`; zero GT uses
`undefined_no_ground_truth`. JSON does not encode nonstandard infinity values.

Repeated part construction runs in a fresh temporary child of the processed
scene. Completed outputs replace `parts` and retain the prior Chinese log; a
failed build does not replace the existing part directory. Registration-based
component removal and original input/output contracts are unchanged.

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

V3's default `run_pipeline.py` and shell runner use the code registry in
`pipeline_components/__init__.py`. `FUSION` exports original PLY/name/256D/384D
files and updated local-to-global frame mappings. `ASSOCIATION` supplies
visibility checks and Hungarian assignment with explicit unmatched observations.
Its part fusion accumulates ownership evidence across frames and permits an
initially unknown parent to become known. `GRAPH_COMPONENTS` publishes official
VLPart predictions into canonical `topology_map.json`; `scene_graph.nodes` contains
objects and confirmed parts, and `scene_graph.edges` contains spatial and hierarchy
relations. The old object fields remain compatible. `TopologyMap.get_parts()` and
`get_entity()` expose the added structure to downstream code.

Detach modules by removing imports and entries from this registry, then rebuild
in a fresh output directory. No backup restore or runtime feature flag is required.
Removing the fusion entry invokes the original C++ implementation. The standalone
part CLI remains available and preserves its input graph.

`object_tracks.json` records distinct frames, label votes and mean observed
confidence. `object_association_zh.jsonl` records geometric/semantic/projected
evidence. `pipeline_zh.jsonl` records construction stages and failures. Results
are separate from input manifests; logs are not consumed as prediction inputs.

The following paragraph documents the retained original binary interface:

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

V3 has two independent roots: `partaware_ai_001_002_v3` and
`partaware_ai_001_010_v3`. New Hypersim preparation uses the full ordered available
frame list, selecting `[::3][:150]`, rather than selecting 150 candidates first.
Official HDF5 sources are retained under each prepared input's `source_hdf5/`.
An existing RGB/depth resolution disagreement is recorded and excluded, never
silently resized or assigned an invented camera registration.

`evaluate_hypersim.py` reads official labels and mesh boxes separately from
construction. Node rank retrieval follows OP3DSG's CLIP ViT-B/16 text protocol,
adapted to fixed NYU40 labels. Predicted boxes come from saved graph shapes;
official oriented mesh boxes are converted to axis-aligned bounds. Category-free
box AP and one-to-one geometric precision/recall are supplementary adaptations,
not ScanNet mesh-mask AP. Voxel occupancy is a density-sensitive diagnostic only.
Part and relation scores remain null when independent labels are unavailable.

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
The retired standalone visualization wrapper has been removed. Use
`script/visualize_map.py` with explicit PLY and topology paths for both layers.

## Original-frontend controls

The existing `validate_pipeline.py` tool now supports two separate controls via
`--frontend-control ram_native|qwen_reuse`, `--manifest`, `--control-scene`, and
`--output`. The control scene must follow `output_root/dataset/scene_id` and be
fresh. These controls run DINO/SAM at confidence 0.4, the original C++ fusion,
spatial graph construction, and point-cloud cleanup. They do not invoke the
registered default fusion, Florence, association, or part components. The same
declared Hypersim floor guards remain active.

`ram_native` passes `--native-vocabulary` to the existing RAM folder/manifest
frontend, preserving the pretrained RAM++ 4585 labels and calibrated thresholds.
Without this option, the original custom-description behavior is unchanged.
The upstream default `scannet509.json` description file is absent from this
checkout; native RAM is a route comparison, not a full paper configuration reproduction.

`qwen_reuse` additionally requires `--reuse-tags-from <completed_scene>`. It
verifies the source graph's manifest hash, copies each frame's tag JSON byte for
byte, and records its SHA-256; it does not invoke the Qwen API or reuse source
segmentation masks. Both controls recompute DINO/SAM observations.

`--control-start-stage segmentation|fusion|graph` resumes an existing control.
Its saved tag source and manifest hash must match. `control_record.json` and
Chinese `pipeline_zh.jsonl` identify the route; `benchmark_provenance` is added
without replacing original object fields. Legacy controls legitimately lack
`object_tracks.json`; default registered-fusion validation still requires it.

The completed second-scene controls are stored under
`scannet-sg-processed/ai_001_010_ram_original_v1` and
`scannet-sg-processed/ai_001_010_qwen_dino_v1`. Each contains the normal
`hypersim/ai_001_010` scene outputs and a root `evaluation.json`. Reproduction
commands must use new output roots; section 15 of `RESEARCH_LOG.html` records the
four-route comparison and common visualization commands.
