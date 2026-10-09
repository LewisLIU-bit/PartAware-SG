# PartAware-SG interfaces

The default Hypersim path applies multi-frame background-supported floors and
low-platform filtering to declared meter-scale Z-up inputs. `floor_filter.json`
records `levels_m`, support and a 1 cm removal band. Background exports retain
cleared points. The legacy C++ fallback retains its original single-floor rule.
Other world-axis conventions are not assumed.

The frontend augments cached Florence/DINO/SAM observations with official YOLOE
masks. When the YOLOE frontend runs, its visual object embedding is a 256D projected DINO backbone ROI
feature, not the older decoder query feature. JSON dimensions are unchanged, but
features from these two spaces must not be averaged or directly compared.
Part features remain independent 1024D RN50 features.

`FRONTEND`, `INSTANCE_REFINEMENT`, `GEOMETRY_COMPONENTS`, `GEOMETRY_OUTPUT`, `OBJECT_VALIDATION` and `BACKGROUND_VALIDATION`
register frontend union, instance consensus, completion and geometry publishing.
Removing imports and entries detaches them; no feature flags or backups are needed.
The completion component skips inputs without verified metric Z-up axes or observed
track provenance, so the original ScanNet and C++ fallback contracts still work.
`build_legacy_graph` preserves the original C++ graph generation path for controls
with the shared cleanup policy. Historical control scores refer to their saved outputs.

`frontend_cache/<id>.npz` stores original observations and RGB hashes for safe reuse;
`frontend_provenance.json` records model/vocabulary signatures and actual candidate
counts. `cache_reuse.json` records reused category hashes and zero Qwen API calls.
The default main path contains no Qwen image request; absent caches use local tags.

`object_validation.json` records candidate rejection and inclusion merges.
`validated_object_tracks.json` provides the accepted observation histories to
completion and evaluation; raw `object_tracks.json` is retained for reproducible
graph-stage recovery. Frame-local mappings and observed PLY IDs are synchronized.
The semantic validator uses RN50 masked crops, not Alpha-CLIP, and no Qwen calls.

`completion_audit.json` records eligibility, model hashes, alignment and all rejection
reasons. `instance_cloud_completed.ply` exists only when a candidate is accepted.
Generated completion points are never written to observed `instance_cloud_cleaned.ply`.
V7 may add depth-verified measured sink part points to that observed file. Generated points are excluded
from association, tracking confidence and observation counts. `topology_map_observed.json`
is the measured-only geometry hypothesis; canonical `topology_map.json` uses measured
or accepted completed geometry and recomputes spatial edges from its box centers.

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
with 256-dimensional visual embeddings and 384-dimensional SBERT text
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

V7's default `run_pipeline.py` uses the code registry in
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

`object_tracks.json` records distinct frames, label votes, reprojection support and quality-weighted observed
confidence (uncalibrated). `object_association_zh.jsonl` records geometric/semantic/projected
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

The current v7 has two independent roots: `partaware_ai_001_002_v7` and
`partaware_ai_001_010_v7`. The formal v6 controls are retained. The v3 inputs and historical outputs are retained. New Hypersim preparation uses the full ordered available
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
commands must use new output roots; the version chapters of `RESEARCH_LOG.html` records the
four-route comparison and common visualization commands.


## Independent AP75 audit

The evaluator now exports `geometry_only_box_AP75`, alongside AP25/AP50,
and one-to-one localization at IoU 0.75. Prediction records include exact
canonical graph box bounds for diagnosis. This does not change masks,
instance construction, graph geometry or the existing evaluation protocol.
The requested AP75 audit covers v5 and the saved RAM/Qwen-DINO controls;
v1-v4 historical scores are not recomputed. Future runs include AP75.

## V6 pointwise held-out-view background validation

`BACKGROUND_VALIDATION` is the sole new algorithm attachment. `background_consensus.py` checks existing candidates over 3 m against up to 100 cached views, excluding their source observation frames. A point requires at least three depth-consistent held-out observations. Rejection requires at least 50% supported points, a strict majority of those points with at least 80% weighted background, and five background-dominant held-out views. Unknown/occluded views do not vote. This is a project adaptation of multi-view mask verification, not a full MaskClustering or Open3DIS reproduction.

`proposal_validation.py` records `direct_background_consensus` and rejection reasons, then regenerates normal PLY IDs, frame mappings, graph boxes/edges and parts. No input depth or completed graph is patched. Raw fusion and background exports remain available. Confidence and prior gates are unchanged. Delete the registry entry/import to detach this component; no backup or runtime flag is required.

The viewer decodes unchanged base-255 RGB IDs across all channels. `--show_edges` controls spatial and part-of links; provided commands omit it and `--enable_picking`. The kitchen opening is not a mirror. The failed combined experiment lives under `_v6_trial`, with `experiment_status.json`; its patch is `/home/lewisliu/datasets/scannet-sg-processed/v6_algorithm_trial.patch`. Those results are not prediction inputs.

## V7 identity, measured part geometry and evaluation diagnostics

Four independent entries attach the new components in `pipeline_components/__init__.py`:

| Entry | Attachment and contract |
| --- | --- |
| `IDENTITY_VALIDATION` | Extends proposal validation for conflicting duplicate names; preserves the accepted object's embeddings and adds `object_identity_aliases` with source IDs, names and evidence. Part vocabulary uses accepted parent identity. |
| `SURFACE_VALIDATION` | Rejects planar sink claims only with held-out depth-consistent background/countertop consensus; unsupported axes and occluded views abstain. It does not classify specular materials. |
| `PART_GEOMETRY` | Runs after `GRAPH_COMPONENTS`; measured basin/faucet/drain points need three supporting part-mask views and seed connectivity. Republishes observed PLY, canonical boxes and spatial edges, preserving other objects' accepted completion geometry. |
| `EVALUATION_DIAGNOSTICS` | Evaluation only: adds `granularity_surface_diagnostic`, without changing strict AP, matching or construction. |

Remove the relevant import and registration entry, then rebuild from raw fusion
observations in a fresh output root. No runtime switch or backup is required.
`part_geometry_audit.json` records component SHA-256, generated point count (zero),
part IDs, candidate/accepted/published point counts and frame evidence. All augmented
points remain measurements. The original object IDs and feature dimensions remain
compatible. Measured sink augmentation currently requires declared Hypersim metric
Z-up input; unknown-axis ScanNet captures keep their original generic path.

The surface diagnostic uses observed 1 cm voxel centers and a 2 cm distance
tolerance. `surface_precision`, `weighted_geometric_surface_coverage`,
`macro_geometric_surface_coverage`, `geometric_surface_F1` and
`geometric_gt_coverage_recall50/75` ignore identity and inspect geometric completeness.
They cannot prove correct semantics or ownership. `weighted_surface_coverage`,
`macro_surface_coverage`, `surface_F1` and `gt_coverage_recall50/75` instead require
dominant GT-assignment purity >= 0.8; `pure_fragment_excess` counts redundant pure
object fragments. GT grouping is evaluation only and is never written into
predicted parent links. Mixed-instance geometry may have high geometric coverage
but fail the pure-fragment test. Coverage recall is not AP. `part_PQ` remains null
because Hypersim does not annotate parts. Registered `part_nodes` already remain
excluded from object AP and object-count metrics.

## V8 measured recovery and whole-object attachments

`MEASURED_REFINEMENT = [thin_geometry, axial_assembly]` runs after
`PART_GEOMETRY`. Each component preserves the existing object schema, base-255
PLY identity encoding, 256D visual and 384D text embeddings. Remove its import and
list item to detach it, then rebuild from raw fusion into a fresh result root.

- `thin_geometry.construct(context)` recovers under-resolved accepted objects
  from cached masks and dense RGB-D with at least three distinct supporting views.
  Automatic 3D point/derived ROI prompts use the existing SAM model. Other accepted
  object arrays are immutable. `thin_geometry_audit.json` records component SHA,
  candidate/accepted counts, view evidence and zero synthetic points.
- `axial_assembly.construct(context)` requires declared metric Z-up, unique
  geometric attachment and at least three jointly supporting ownership views.
  Axial supports require stem/base shape and alignment. Upper crossbars require
  an accepted recovered parent, upper position, axis agreement and measured contact;
  all input views use jointly registered strict depth/mask validation. Multiple
  crossbars may attach to one parent, while each crossbar has one eligible parent.
- Source coordinates are concatenated without resampling; geometric `body`,
  `support` and `crossbar` parts include `source_object_id`, `point_count`,
  `observed_frames`, `semantic_embedding: null`, and
  `semantic_feature_space: geometry_only_no_visual_embedding`. Their point files
  are `parts/assembly_<source>_<role>.points.npy`. They are not visual VLPart embeddings.
- `object_identity_aliases` retains source identity; updated frame mappings and
  existing part parents follow the canonical body. `assembly_audit.json` records
  contact/axis measurements, source counts and supporting frames. Whole-object
  confidence and embeddings remain those of the original accepted body.
- `topology_map.json`, `topology_map_observed.json`, canonical PLY and
  `parts/partaware_graph.json` are republished together. Evaluation uses the same
  strict one-to-one object protocol, with parts excluded. No GT is read during
  either construction component. Static contact is evidence for an assembly
  hypothesis, not proof of mechanical rigidity or articulations.

## V9 observed evidence, bounded cuboids and MVO

- `OBSERVED_VALIDATION` validates jointly registered depth and local mask ownership. Stable measured-core rescue applies only to semantic-only rejection; mixed/held-out-background/sink vetoes remain independent. `object_validation.json` stores per-view evidence and core bounds.
- `GEOMETRY_COMPONENTS` appends `backed_cuboid` after AdaPoinTr. It uses an observed front, side support, parallel measured rear plane and all-view free-space checks. Hypotheses carry `geometry_hypothesis.measured=false`; `cuboid_completion_audit.json` keeps acceptance assumptions. It never mutates the measured PLY or supplies association evidence.
- `MEASURED_REFINEMENT` appends `assembly_density` after axial assembly. Only accepted crossbars receive verified dense RGB-D points. Corresponding part geometry, tracks, canonical boxes and edges are republished.
- Detach each component by removing its import/registry entry, then rebuild from raw observations in a fresh result root. No backup/feature flag is needed.
- `maximum_volume_overlap` in new evaluations includes MVO AP25/AP50 and one-to-one TP/FP/FN. Historical `evaluation_mvo.json` files reference frozen evaluation hashes and fixed boxes. MVO uses strictly greater thresholds; legacy IoU AP still uses greater-or-equal thresholds.
- The saved graph remains the same object/edge/part interface. Generated geometry belongs only in `instance_cloud_completed.ply`, and `topology_map_observed.json` remains measured-only.

## V10 bounded measured structures and original-pixel suspension members

GEOMETRY_COMPONENTS places structural_surfaces before completion. FINAL_GEOMETRY
contains suspension_geometry after measured assembly refinements. Both require
declared metric Z-up; other inputs preserve the original fallback. New nodes
retain the original id/name/position/shape/256D/384D contract. Only rejected local
observation IDs may be remapped; original PNG/feature/name caches remain fixed.
The first component checks cached compatible identities, measured front/rear
boundaries and three-view depth support. The second projects automatically
derived body-to-ceiling ROIs at stride 1 and 3 mm voxel size, first recovers seed-connected measured upper-body surfaces within 25 cm of the
existing facet. Separate rod candidates require PCA line shape, at least 10 of
12 occupied height bins, unique contact and three-view depth support; endpoints
alone cannot establish a shaft. The current final run accepts upper-body geometry
and zero continuous rods.
Other accepted measured objects are protected and every original point remains.

structural_surface_audit.json and suspension_geometry_audit.json store geometric
acceptance evidence, source identities, measured additions, zero generated points
and zero Qwen calls. Suspension parts carry no visual feature and use geometry-only
provenance. Accepted generated cuboids remain distinct in the completed PLY.

Final canonical publication runs after all components, synchronizing boxes, edges,
scene_graph and parts/partaware_graph.json. --start-stage final_geometry resumes
the complete recovery stage; --start-stage publish only resumes publication.
Component removal still requires deleting its import and registry entry, then
rebuilding a fresh result from raw observations, without feature flags or backups.
The original viewer accepts optional --view-front, --view-lookat, --view-zoom.
These change the camera only; edges and picking remain opt-in.

Closed storage identity is inferred before part construction from cached cabinet/cupboard observations in at least three distinct frames, facade grid occupancy >= 0.60, at least 128 measured side points and side extension to half the independently measured depth. The existing depth/baseline checks still apply. storage_identity records the evidence separately; cached labels and 256D/384D features are not rewritten. Open racks and sheets do not gain a closed cabinet label. Whole-cabinet mask-tree grouping remains a research proposal.

## V11 instance granularity and per-object diagnostics

`GEOMETRY_COMPONENTS` runs `instance_granularity` before structural recovery. The adapter reads accepted measured geometry, cached masks and validated tracks. Unique shared-surface duplicates retain the largest parent ID, name, feature vectors and confidence. All source coordinates are concatenated, local ownership and aliases are resolved transitively, and tracks record `granularity_source_ids`. It writes `instance_granularity_audit.json`; no GT or language request enters construction. Independently gated region transfer never creates a new node and was not accepted in the current runs. Detach by removing its import and list entry, then rebuild from raw caches in a new result directory.

`object_error_diagnostic` is evaluation only. `per_gt` stores the best three predictions, `one_to_one_matches` at 0.25/0.5/0.75 and surface diagnostics. `per_prediction` and `AP_ranking` distinguish matched objects, duplicate competition and localization below threshold. Ranking follows the existing greedy AP rule; optimal one-to-one counts can differ. These fields do not alter boxes, scores or GT.

## V12 native masks and measured body hierarchy

`SURFACE_ASSEMBLY` registers `native_assembly.reconcile` only in v12. It resolves
signed complete-mask cache provenance from `cache_reuse.json`, queries packed
original masks, and returns canonical measured candidates before publication.

The v12 `MEASURED_REFINEMENT` list runs `part_body_assembly.construct` after thin
recovery. BHA accepts unique measured boundary ownership and creates queryable
geometry-only panel parts with null visual embeddings. It synchronizes measured
PLY, validated tracks, frame IDs, aliases, parts, hierarchy edges and canonical
spatial geometry. It cannot infer door semantics or concealed cabinet depth.
Detach either component by deleting its import and registry entry, then rebuild
from raw observations. Current receipts and unchanged-GT metrics are in
`docs/sam3_woc_results.json` and report chapter 22. Object AP excludes parts;
part AP remains unavailable without part annotations.
