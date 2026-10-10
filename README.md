# PartAware-SG (based on ScanNet-SG)

This repository contains the code for the __ScanNet-SG__ Dataset.

This dataset is built on top of ScanNet by adding 3D scene graphs that contain open-set visual-language (GroundingDINO) features, BERT features, bounding boxes, etc., for each object in each scene.
The dataset is mainly designed for frame-to-scan and subscan-to-subscan scene graph alignment. But it can also be used for the validation of navigation.

For more details, please refer to our paper:
__ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment__ and 
__OpenSGA: Efficient 3D Scene Graph Alignment in the Open World__ (Coming soon).


## Default PartAware-SG pipeline

GPT category recognition supports an OpenAI-compatible relay through
`scannet/script/vision_api.py`. Each existing scene has one complete GPT cache
for v12, v13 and v14; construction makes no new GPT or Qwen image requests. Report defaults
remain original RAM, existing Qwen for v11, and existing GPT for v12 and later. The GPT original pipeline is retained as a separate baseline and compared
with RAM, Qwen-to-DINO, v12, v13 and v14 using the same scene cache as v12. See
[GPT setup, direct commands and comparison protocol](docs/VISION_API.md).
The existing ScanNet and Hypersim interfaces remain unchanged.

The latest v14 research profile reuses the existing GPT and SAM3 observations.
Its FOVEA -> MICA -> SHAPE -> GRAPH functional tree now includes identity-seeded
measured surface proposals, closed-front versus open-rack boundaries, physical
assembly ownership checks, and all-view verified shape hypotheses after assembly.
Names select a bounded geometric prior; measured faces determine its dimensions.
Generated surfaces remain hypotheses and never replace measured coordinates.

Both scenes are rebuilt and independently evaluated. ai_001_002 retains
100%/100%/100% AP25/AP50/AP75; ai_001_010 reaches 44.42%/27.68%/10.86%
from v13's 42.02%/25.97%/9.91%. The 80% AP25 / 70% AP50 target remains unmet.
These are class-agnostic adapted Hypersim box metrics. Universal text-conditioned
completion and broad fine-instance generalization remain unresolved; official MGPC
inference was tested but did not pass the observation constraints and is not active.
See [the consolidated report](docs/RESEARCH_REPORT.md) for final per-object results,
GT explanations, equations, source papers and direct visualization commands.

To rebuild v14 from cached raw fusion, choose a fresh result path:

```bash
cd /home/lewisliu/PartAware-SG
/home/lewisliu/miniconda3/envs/scannet-sg/bin/python scannet/script/run_gpt_comparison.py \
  --worker v14 \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_010_v3/manifest.json \
  --reuse-scene /home/lewisliu/datasets/scannet-sg-processed/partaware_v13/hypersim/ai_001_010 \
  --processed-scene /home/lewisliu/.cache/partaware-sg/rebuild_v14/hypersim/ai_001_010 \
  --start-stage graph
```

Final artifacts are under `datasets/scannet-sg-processed/partaware_v14`.
Pair `instance_cloud_completed.ply` with `topology_map.json`, or use
`instance_cloud_cleaned.ply` and `topology_map_observed.json` for measurements.
The first scene has no added completion and uses the cleaned PLY.
Detach adapters at the code registrations in `configure_profile`, then rebuild;
no algorithm flags or backup restoration are needed. Historical versions and
ordinary ScanNet/Hypersim interfaces remain available.

PartAware-SG preserves ScanNet folders and Hypersim manifests. Its v11 default
constructs objects, parts and hierarchy in one canonical `topology_map.json`,
using [OP3DSG](https://github.com/AutoCompSysLab/OP3DSG)-inspired fusion,
visibility-aware one-to-one association and official
[VLPart](https://github.com/facebookresearch/VLPart) inference.
The original object JSON format, 256-dimensional visual features, and
384-dimensional text features are preserved. Basic observations use GroundingDINO proposals, local Florence crop evidence
and SAM masks. V4 augments them with official YOLOE-v8-S segmentation, then applies
multi-view instance consensus, observed-surface refinement and selective official
AdaPoinTr-PCN completion before publishing geometry. V5 inserts observed proposal validation and masked-CLIP
SMS filtering before completion; rejected candidates do not enter the canonical
objects or their cleaned PLY. Logs retain all decisions and original fusion tracks. Generated surfaces never count
as observed evidence. When the YOLOE frontend runs, object features use a consistent 256D projected
DINO backbone ROI space; these differ from the older 256D decoder query features. ScanNet folders and Hypersim manifests keep
their original interfaces; `--grounding_backend dino` restores the original
frontend, or `GROUNDING_BACKEND=dino` selects it in the shell runner. The retained legacy folder frontend keeps its original decoder features; do not
mix them with ROI features inside one capture. Florence
does not replace the 256-dimensional detector features or infer physical instance
identity. `--joint_config` still accepts the existing scene-specific reference
profiles. The portable default scores all input categories with soft evidence
and needs no scene-specific reference images. See setup for the isolated runtime.
The default runner now uses code-registered construction components. The original
`openset_ply_map` binary and its `legacy` default remain available as the fallback
when the fusion component is detached. This is an adapted prior-graph pipeline;
OP3DSG's LLM reasoning stage is not included.

V7 resolves conflicting duplicate identities only with contained observed surfaces
and independent complete-mask evidence. A separate planar-surface validator rejects
sink claims contradicted by held-out background or countertop observations. After
VLPart fusion, measured sink component points require three depth-consistent part
mask views before augmenting object geometry; boxes and spatial edges are then
republished through the same interface. This is not a general specular-material
classifier. No ground truth or new Qwen image call enters construction.
Strict AP25/AP50/AP75 remain unchanged. Additional surface coverage and pure-fragment
diagnostics distinguish geometric completeness from instance ownership; they do
not replace AP or supply unavailable part ground truth. See the current mathematics
and version-specific results in [the research report](docs/RESEARCH_REPORT.md).

V8 adds measured thin-surface recovery with dense cached masks, three-view depth
consensus and automatic projected 3D SAM point/ROI prompts. It uses no new language
prompts or Qwen calls. Independent unrelated point arrays are preserved exactly.
A second component groups uniquely attached axial supports and upper crossbars,
retains every source coordinate as a geometric part, and republishes the same
object boxes, hierarchy and spatial edges. Strict AP and confidence sorting are
unchanged. Detach either component by removing its import/list item from
`MEASURED_REFINEMENT`; rebuild from raw observations in a fresh directory.
No additional model or dependency installation is required. These are adaptations
of SAMPro3D/SAI3D principles, not complete reproductions. Recovery adds only
observed RGB-D points; unsupported hanging rods are not fabricated.

The default path uses joint registered depth/mask evidence and stable measured-core rescue for
semantic-only rejections. Mixed-instance and background vetoes remain active. An
independent geometric cuboid component uses an observed front, side support and a
parallel measured rear structure; all capture views check free space and foreign
objects. Accepted inferred surfaces are stored separately and affect the common
graph boxes. This is not a learned refrigerator/CAD model. Verified crossbar
measurements are densified after attachment, preserving unrelated geometry.
Remove `OBSERVED_VALIDATION`, the `backed_cuboid` geometry entry or the
`assembly_density` measured-refinement entry to detach them independently.
The evaluator adds user-defined MVO = intersection / max(box volumes), with
strict >25% and >50% thresholds, while original IoU AP is unchanged.

Declared meter-scale Hypersim Z-up inputs use supported floor and low-platform
filtering before association. `floor_filter.json` records accepted heights and
multi-view background evidence. Other coordinate conventions keep the original
generic geometry path. Final boxes and spatial edges use the published cleaned
or accepted completed cloud, without filtering it a second time. An independent
`topology_map_observed.json` retains measured-only boxes for completion evaluation.

Only ScanNet and Hypersim are supported by the project-specific adapters.
Run artifacts are stored outside the source tree, under `datasets/scannet-sg-processed/<name>_v1`, `<name>_v2` or `<name>_v3`.
The current WSL copy already contains the required runtime repositories, weights, and
isolated part environment. Use `scannet-sg` for the object frontend and
`.venv-vlpart/bin/python` for parts; YOLOE and completion use
`.venv-yoloe` and `.venv-completion` in the project. See [reproducible setup](docs/PARTAWARE_SETUP.md)
for source revisions, dependencies, and official weight downloads.

```bash
cd /home/lewisliu/PartAware-SG
conda activate scannet-sg
cmake -S scannet -B scannet/build-partaware
cmake --build scannet/build-partaware -j2
# Use a new experiment name; the basic frontend defaults to Florence.
MAX_FRAMES=3 bash run_scannet_sg.sh scene0000_00 scannet_partaware_v3
```

Hypersim's full default pipeline reuses completed same-scene observations and
cached Qwen category descriptions. It does **not call Qwen image recognition**.
Missing category caches use the fixed indoor vocabulary locally. For the existing
two scenes, use a fresh result directory and copy the prior observations:

```bash
python scannet/script/run_pipeline.py \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_010_v3/manifest.json \
  --reuse-scene /home/lewisliu/datasets/scannet-sg-processed/partaware_v5/hypersim/ai_001_010 \
  --processed-scene /home/lewisliu/datasets/scannet-sg-processed/my_hypersim_v10/hypersim/ai_001_010 \
  --start-stage graph
```

For a new scene, omit `--reuse-scene`: local vocabulary, Florence/SAM and YOLOE
produce the frontend without API credentials. Prepare its complete ordered frame
list every third frame, capped at 150 selected frames; 300 frames yield 100 inputs:

```bash
python scannet/script/prepare_hypersim.py \
  --scene ai_001_010 \
  --output /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_010_v3 \
  --frame-step 3 --max-frames 150
```

The original ScanNet folder frontend remains in `run_scannet_sg.sh`. Feed its
completed observations to the current construction path with
`run_pipeline.py --image-dir <original_scene> --processed-scene <result_scene>
--start-stage fusion`, preserving original files and IDs. The legacy folder
frontend does not call Qwen.

Official candidate HDF5 files and independent labels stay in `source_hdf5/`.
Only selected RGB-D frames appear in the manifest. No complete ZIP archive or
download helper repository is retained. `--existing-manifest` reuses downloaded
RGB-D without changing it and records dimension-invalid frames explicitly.

Base component attachment is centralized in
`scannet/script/pipeline_components/__init__.py`; research-version adapters are
registered in `run_gpt_comparison.configure_profile`. To detach an algorithm, remove
its import and registry entry: `ASSOCIATION` returns object association and part
ownership to their conservative base rules; `FUSION` returns object fusion to the
original C++ implementation; entries in `GRAPH_COMPONENTS` determine graph
extensions. `FRONTEND`, `INSTANCE_REFINEMENT`,
`GEOMETRY_COMPONENTS`, `GEOMETRY_OUTPUT`, `OBJECT_VALIDATION` and `BACKGROUND_VALIDATION` register the current additions. Detachment uses code editing, not feature flags or backup restoration.
Rebuild into a fresh result folder after editing the registry. Existing result
files do not change automatically. `--start-stage` resumes a completed frontend;
it is not a component-removal switch.

V6 adds one removable pointwise held-out-view background check to the existing validator. Large mixed identities must agree with independent depth-consistent views; source masks cannot validate themselves. The stable v5 segmentation, association, inclusion and confidence rules remain in use. Results are regenerated from cached raw fusion outputs, not edited by hand; no new Qwen calls or input-depth changes occur. Earlier combined SAVPE/partition experiments were withdrawn; only final version outputs are retained.

The standalone `scannet/script/run_partaware.py` entry remains available for
inspecting an existing graph; its ScanNet and Hypersim contracts are documented
in [interfaces](docs/PARTAWARE_INTERFACES.md). `--sam-checkpoint` optionally refines
boxes with the preserved SAM weight. `--dbscan-eps 0.05`, `--subtract-contained`,
and `--erode-pixels 1` are independent, disabled-by-default cleanup experiments.
Containment cleanup requires the same resolved parent instance, and ambiguous
ownership stays unresolved. `--part-overrides` accepts a JSON mapping from object
names to part names; objects absent from the knowledge base get no guessed parts.

Outputs include `partaware_graph.json`, independent per-frame Boolean masks,
observed part point clouds, `run_config.json`, and Chinese `run_zh.jsonl` logs.
The standalone command preserves its supplied `topology_map.json`; the default
full pipeline publishes the returned parts into canonical `topology_map.json` and
its `scene_graph` extension. Parts have their own 1024-dimensional
CLIP RN50 feature space; confirmed, attached tracks create `part_of` edges.
See [interfaces](docs/PARTAWARE_INTERFACES.md).
The [offline LaTeX-rendered HTML report](RESEARCH_LOG.html) includes all equations
as embedded SVG and opens without runtime downloads. Editable
[LaTeX source](RESEARCH_LOG.tex) and editable
[Chinese research source](docs/RESEARCH_REPORT.md) are also provided. General
flow, mathematics and interfaces describe v10; each version change and the withdrawn v6 trial have separate chapters.

```bash
python -m unittest discover -s scannet/script/tests_partaware -v
```

Earlier ScanNet and Hypersim object regression runs produced identical graphs with
the saved binary and new default binary. Native VLPart and optional SAM runs
completed on small samples. These runs establish functionality and compatibility;
detector errors, duplicate parts, and uncertain ownership remain. V3 uses independent
official Hypersim instance labels and mesh boxes for adapted object evaluation;
Hypersim does not provide part/hierarchy/functional-relation ground truth.
Object evaluation reports class-agnostic 3D box AP25, AP50 and AP75.
It also reports the predicted/annotated count ratio and symmetric
absolute natural-log count error, `abs(log(N_pred / N_annotated))`. Lower is better;
zero predictions with nonzero annotations have infinite error, represented by a
JSON null plus an explicit status. Counts exclude structural background and parts.
Missed and duplicate objects can cancel, so count error complements localization.
These are not official ScanNet or UniGraph3D benchmark scores. See
[the HTML report](RESEARCH_LOG.html) for results and limitations.

```bash
python scannet/script/evaluate_hypersim.py \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_002_v3/manifest.json \
  --processed-scene /home/lewisliu/datasets/scannet-sg-processed/partaware_v11/hypersim/ai_001_002 \
  --output /home/lewisliu/datasets/scannet-sg-processed/partaware_v11/hypersim/ai_001_002/evaluation.json
```


### Visualize the generated part graph

Object and refined part graphs now use the same `visualize_map_with_nodes`
renderer. Without `--show_parts`, the original object geometry is unchanged.
With it, smaller parent-colored child nodes are appended; dashed `part_of` links require `--show_edges`. These overlays are appended to the original point cloud, bounding boxes, relations, and picking.
Unassigned or provisional parts are hidden by default; part point clouds are
also hidden so the original map remains readable.

```bash
env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET \
XDG_SESSION_TYPE=x11 LIBGL_ALWAYS_SOFTWARE=true \
python script/visualize_map.py \
  --map_ply_path /home/lewisliu/datasets/scannet-sg-processed/partaware_v11/hypersim/ai_001_002/instance_cloud_cleaned.ply \
  --topology_map_path /home/lewisliu/datasets/scannet-sg-processed/partaware_v11/hypersim/ai_001_002/topology_map.json \
  --show_bboxes --show_parts \
  --node_radius 0.07 --part_radius 0.025
```

Remove `--show_parts` for the original object view. Add `--show_part_points`
for diagnostic part clouds, `--include_provisional_parts` for all part tracks,
or `--check_only` to validate the same geometry without a window.
Add `--enable_picking` only when Shift + left click should print a node name. Use tracking-ID PLY files
(`instance_cloud*.ply`), rather than already recolored RGB exports.
The current second-scene experiment is under
`/home/lewisliu/datasets/scannet-sg-processed/partaware_v11/hypersim/ai_001_010`.
Use `instance_cloud_completed.ply` to view its accepted inferred cuboid surfaces;
use the cleaned PLY with `topology_map_observed.json` for measured-only geometry.
Use that directory for both visualization paths. For its dense spatial graph,
omit `--show_edges` when inspecting objects and part ownership.

The second-scene frontend controls are stored separately as
`ai_001_010_ram_original_v1` and `ai_001_010_qwen_dino_v1` under
`datasets/scannet-sg-processed`. Both recompute DINO/SAM and use the original
C++ graph path. The Qwen control reuses cached tags without an API call.
See [control interfaces](docs/PARTAWARE_INTERFACES.md#original-frontend-controls)
and [version chapters of the report](RESEARCH_LOG.html)
for the four-route AP25/AP50/count comparison and common visualization commands.

V10 restores bounded measured cabinet/countertop surfaces from cached identities
and original depth evidence, resolves closed storage identity from cached category votes plus measured facade/side evidence, then recovers measured upper fixture bodies before final canonical publication.
Thin member proposals additionally require shaft continuity; aligned endpoints
are insufficient, and zero full rods pass the current scene verification. No new Qwen image request is made. Geometry-only
parts remain separate from the VLPart RN50 feature space. Detach the two recovery
components through their code registry entries. Optional --view-front,
--view-lookat and --view-zoom affect only the common viewer camera.
See the current mathematics and measured limitations in the research report.

## Environment Installation

Use the existing WSL environments for normal runs. Installation and model setup
are documented once in [reproducible setup](docs/PARTAWARE_SETUP.md).
Original baseline installation is described in the
[upstream ScanNet-SG README](https://github.com/tud-amr/ScanNet-SG#environment-installation).
Build the required C++ tools with the commands in the default pipeline example above.


## Map Interface Usage

### Python version:
The Python class for I/O of the SceneGraph (or TopologyMap) JSON file is defined in `script/include/topology_map.py` as `TopologyMap`.
Check the examples below to learn how to use the interface. (The interface also contains free-space nodes, but we do not use them in the current version.)

- Visualize a scene graph using the following command
```bash
env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET \
XDG_SESSION_TYPE=x11 LIBGL_ALWAYS_SOFTWARE=true \
python script/visualize_map.py --show_bboxes
```
Add `--map_ply_path xxx.ply --topology_map_path xxx.json` to specify the data. Add `--enable_picking` to use interactive mode: the name of a node will be printed when you press `Shift` and left-click the blue sphere of a node.
By default, example data in `sample_data/scans/scene0000_00` will be used. You will see an image like the following:

![image](/sample_data/scans/scene0000_00/scene_0000.png)



### C++ version:

C++ data structure is defined in `include/topology_map.h` 
Check the example in the following to know how to use the C++ version interface.

The C++ map structures remain in `include/topology_map.h` and are used by the
retained `generate_json` target.


## Generate Scene Graphs with Your Own Data
Use the ScanNet and Hypersim commands in the default pipeline section.
Input/output formats are documented in [interfaces](docs/PARTAWARE_INTERFACES.md).
Retired batch alignment, subscan generation, ROS examples, and duplicate viewers
have been removed; the retained entry points and checks are recorded in
[the Chinese cleanup log](docs/CLEANUP_LOG.md).


## Citation
```
@dataset{scannet_sg,
  author    = {Gang Chen and Sebastián Barbas Laina and Javier Alonso-Mora},
  title     = {ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment},
  year      = {2026},
  doi       = {10.4121/bebe8bd4-cf91-4f86-a28a-87cb870f6cea}, 
  url       = {https://data.4tu.nl/datasets/bebe8bd4-cf91-4f86-a28a-87cb870f6cea}
}
```

## Licence
The code in this repo uses the Apache-2.0 licence.
The dataset uses CC BY-NC 4.0 licence.

### V11 instance granularity and error diagnostics

The default graph stage now resolves unique shared-surface duplicates before structural recovery and completion. It requires contained measured points, semantic compatibility, at least five complete-mask consensus views and no independent separation evidence. Existing coordinates and public graph fields remain intact; `instance_granularity_audit.json` records the decisions. Detached-region transfer is independently gated and no transfer was accepted in the two current scenes. Remove the import and `GEOMETRY_COMPONENTS` entry to detach this adapter.

The evaluator adds `object_error_diagnostic`, with per-GT best IoU, optimal one-to-one matches and the actual AP confidence-ranking assignments. These diagnostics never change AP or GT. The latest two-scene results and direct visualization commands are in [the consolidated report](docs/RESEARCH_REPORT.md).

### GPT cache compatibility and measured ownership refinement

The historical GPT comparison profiles validate visible surface ownership and measured floor boundaries, resolve a unique full-object anchor among contained fragments, and trim contradicted fringes only around a verified observed core. Fine instances and independently visible objects remain protected. This changes construction from cached observations; it does not delete selected final nodes or infer hidden dimensions. Detach it at the `OWNERSHIP_VALIDATION` code registration in `run_gpt_comparison.configure_profile`. Current final metrics are reported for the SAM3-based v12 below; superseded intermediate results have been removed. Mathematical details and actual results are in [the consolidated Chinese report](docs/RESEARCH_REPORT.md).

### V12 GPT cache and SAM3 observation backend

Default recognition sources in the report are RAM for the original baseline, existing Qwen caches for v11, and the existing shared GPT cache for v12. Further GPT v11 tuning is stopped. New construction uses the v12 profile and refuses incomplete recognition caches instead of making a new image API request. Retained GPT original results participate in the final baseline and v12 comparison.

The v12 research registration now uses `pipeline_components.sam3_frontend`: local SAM3 concept masks and adaptive crops feed the retained uint8 mask, 256D DINO and 384D SBERT interfaces. SAM3 weights are downloaded from [ModelScope](https://modelscope.cn/models/facebook/sam3) and checked against its SHA256; inference uses the pinned [Meta source](https://github.com/facebookresearch/sam3) in the managed WSL Conda environment `sg-sam3`. Both real single-frame smoke tests pass. After measured whole-object consensus revisions, the first full scene retains 10 objects, AP25/AP50 stay at 100%, and AP75 reaches 100%; the second full scene improves AP25/AP50/AP75 to 32.21%/18.86%/6.17%, while the strict all-metric gate remains unpassed. SAM3 does not generate unobserved 3D surfaces. Change the v12 `FRONTEND` code registration back to `fovea` to detach this backend. See [the interface and direct commands](docs/VISION_API.md) and report chapters 18–21.

### V12 whole-object surface consensus

`whole_object_consensus.reconcile` reuses existing masked RN50 identity features, measured RGB-D surfaces and complete-mask view evidence after proposal ownership validation. Shared-surface duplicates, differently named complementary bodies and a dominant measured terminal face require mutually unique ownership; independently separated objects remain distinct. Robust planar fitting supports elongated faces with minority edge returns. No hidden points or ground-truth dimensions are generated. Remove the v12 `WHOLE_OBJECT_VALIDATION` import/registration to detach this adapter. The first scene passes strict metric acceptance and publication checks; [the consolidated report](docs/RESEARCH_REPORT.md) records final results only. The full 178-test offline suite passes. Latest metrics and artifact hashes are in [the result receipt](docs/sam3_woc_results.json).
### V12 measured body hierarchy

`native_assembly.reconcile` queries signed original SAM3 masks rather than only
the exclusive public label image. Repeated whole-mask support and measured
contact can associate fragments, while separate small dishes remain protected.
Detach the v12 `SURFACE_ASSEMBLY` import/registration to remove this adapter.

`part_body_assembly.construct` adds BHA (Body Hierarchy Association) after thin
geometry recovery and before axial assembly. A planar measured subinstance
requires a unique volumetric measured anchor, matching identity, an aligned
boundary and at least three common observed frames. The original points remain
queryable as `assembly_<id>_panel` parts with geometry-only provenance; no door
semantics or hidden depth is invented. Delete its v12 `MEASURED_REFINEMENT`
import/registration to detach it. In ai_001_010 it retains 13 measured panels,
reduces standalone objects from 139 to 126, and preserves every other object's
point coordinates. Object AP uses the unchanged GT; part quality has no AP
without part-level annotations. See report sections 2.7/2.8, 6.7 and chapter 18.

Current results share `scannet-sg-processed/partaware_v12/hypersim/`, with
`ai_001_002` and `ai_001_010` as sibling scenes. Direct viewer commands are in
report section 21;
`--show_parts --show_part_points` optionally displays the retained panel parts.
Edges and picking remain opt-in.
