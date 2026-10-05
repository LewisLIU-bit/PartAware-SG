# PartAware-SG (based on ScanNet-SG)

This repository contains the code for the __ScanNet-SG__ Dataset.

This dataset is built on top of ScanNet by adding 3D scene graphs that contain open-set visual-language (GroundingDINO) features, BERT features, bounding boxes, etc., for each object in each scene.
The dataset is mainly designed for frame-to-scan and subscan-to-subscan scene graph alignment. But it can also be used for the validation of navigation.

For more details, please refer to our paper:
__ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment__ and 
__OpenSGA: Efficient 3D Scene Graph Alignment in the Open World__ (Coming soon).


## Default PartAware-SG pipeline

PartAware-SG preserves ScanNet folders and Hypersim manifests. Its v5 default
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
  --reuse-scene /home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_010_v3/hypersim/ai_001_010 \
  --processed-scene /home/lewisliu/datasets/scannet-sg-processed/my_hypersim_v5/hypersim/ai_001_010
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

Component attachment is centralized in
`scannet/script/pipeline_components/__init__.py`. To detach an algorithm, remove
its import and registry entry: `ASSOCIATION` returns object association and part
ownership to their conservative base rules; `FUSION` returns object fusion to the
original C++ implementation; entries in `GRAPH_COMPONENTS` determine graph
extensions. `FRONTEND`, `INSTANCE_REFINEMENT`,
`GEOMETRY_COMPONENTS`, `GEOMETRY_OUTPUT` and `OBJECT_VALIDATION` register the current additions. Detachment uses code editing, not feature flags or backup restoration.
Rebuild into a fresh result folder after editing the registry. Existing result
files do not change automatically. `--start-stage` resumes a completed frontend;
it is not a component-removal switch.

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
flow, mathematics and interfaces describe v5; v1-v5 changes have separate chapters.

```bash
python -m unittest discover -s scannet/script/tests_partaware -v
```

Earlier ScanNet and Hypersim object regression runs produced identical graphs with
the saved binary and new default binary. Native VLPart and optional SAM runs
completed on small samples. These runs establish functionality and compatibility;
detector errors, duplicate parts, and uncertain ownership remain. V3 uses independent
official Hypersim instance labels and mesh boxes for adapted object evaluation;
Hypersim does not provide part/hierarchy/functional-relation ground truth.
Object evaluation also reports the predicted/annotated count ratio and symmetric
absolute natural-log count error, `abs(log(N_pred / N_annotated))`. Lower is better;
zero predictions with nonzero annotations have infinite error, represented by a
JSON null plus an explicit status. Counts exclude structural background and parts.
Missed and duplicate objects can cancel, so count error complements localization.
These are not official ScanNet or UniGraph3D benchmark scores. See
[the HTML report](RESEARCH_LOG.html) for results and limitations.

```bash
python scannet/script/evaluate_hypersim.py \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_002_v3/manifest.json \
  --processed-scene /home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_002_v5/hypersim/ai_001_002 \
  --output /home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_002_v5/evaluation_v5.json
```


### Visualize the generated part graph

Object and refined part graphs now use the same `visualize_map_with_nodes`
renderer. Without `--show_parts`, the original object geometry is unchanged.
With it, smaller parent-colored child nodes and dashed `part_of` links are
appended to the original point cloud, bounding boxes, relations, and picking.
Unassigned or provisional parts are hidden by default; part point clouds are
also hidden so the original map remains readable.

```bash
env -u WAYLAND_DISPLAY -u WAYLAND_SOCKET \
XDG_SESSION_TYPE=x11 LIBGL_ALWAYS_SOFTWARE=true \
python script/visualize_map.py \
  --map_ply_path /home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_002_v5/hypersim/ai_001_002/instance_cloud_cleaned.ply \
  --topology_map_path /home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_002_v5/hypersim/ai_001_002/topology_map.json \
  --show_bboxes --show_edges --show_parts --enable_picking \
  --node_radius 0.07 --part_radius 0.025
```

Remove `--show_parts` for the original object view. Add `--show_part_points`
for diagnostic part clouds, `--include_provisional_parts` for all part tracks,
or `--check_only` to validate the same geometry without a window.
Shift + left click prints either an object or child node name in the same
window. Use tracking-ID PLY files
(`instance_cloud*.ply`), rather than already recolored RGB exports.
The second full v5 experiment is under
`/home/lewisliu/datasets/scannet-sg-processed/partaware_ai_001_010_v5/hypersim/ai_001_010`.
Use that directory for both visualization paths. For its dense spatial graph,
omit `--show_edges` when inspecting objects and part ownership.

The second-scene frontend controls are stored separately as
`ai_001_010_ram_original_v1` and `ai_001_010_qwen_dino_v1` under
`datasets/scannet-sg-processed`. Both recompute DINO/SAM and use the original
C++ graph path. The Qwen control reuses cached tags without an API call.
See [control interfaces](docs/PARTAWARE_INTERFACES.md#original-frontend-controls)
and [version chapters of the report](RESEARCH_LOG.html)
for the four-route AP25/AP50/count comparison and common visualization commands.

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
python script/visualize_map.py --show_bboxes --show_edges
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
