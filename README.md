# PartAware-SG (based on ScanNet-SG)

This repository contains the code for the __ScanNet-SG__ Dataset.

This dataset is built on top of ScanNet by adding 3D scene graphs that contain open-set visual-language (GroundingDINO) features, BERT features, bounding boxes, etc., for each object in each scene.
The dataset is mainly designed for frame-to-scan and subscan-to-subscan scene graph alignment. But it can also be used for the validation of navigation.

For more details, please refer to our paper:
__ScanNet-SG: A Large-Scale Dataset for 3D Scene Graph Alignment__ and 
__OpenSGA: Efficient 3D Scene Graph Alignment in the Open World__ (Coming soon).


## Optional PartAware-SG pipeline

PartAware-SG preserves the saved ScanNet and Hypersim object pipelines and adds
an independent object-part layer using official [OP3DSG](https://github.com/AutoCompSysLab/OP3DSG)
knowledge and [VLPart](https://github.com/facebookresearch/VLPart) inference.
The original object JSON format, 256-dimensional visual features, and
384-dimensional text features are preserved. The default object association
remains `legacy`; append `--association_mode op3dsg` to `openset_ply_map` only
for a separate association experiment. This is an adapted prior-graph pipeline;
OP3DSG's LLM reasoning stage is not included.

Only ScanNet and Hypersim are supported by the project-specific adapters.
Run artifacts are stored outside the source tree, under `datasets/scannet-sg-processed/<name>_v1`.
The current WSL copy already contains the required runtime repositories, weights, and
isolated part environment. Use `scannet-sg` for the object frontend and
`.venv-vlpart/bin/python` for parts. See [reproducible setup](docs/PARTAWARE_SETUP.md)
for source revisions, dependencies, and official weight downloads. The upstream
installation instructions below remain available for the original pipeline.

```bash
cd /home/lewisliu/PartAware-SG
conda activate scannet-sg
cmake -S scannet -B scannet/build-partaware
cmake --build scannet/build-partaware -j2
# Use a new experiment name for an original ScanNet run.
MAX_FRAMES=3 bash run_scannet_sg.sh scene0000_00 scannet_baseline_v1
```

After an object graph has been generated, run parts in a fresh output directory.
This verified ScanNet example uses the existing object baseline:

```bash
.venv-vlpart/bin/python scannet/script/run_partaware.py \
  --image-dir /home/lewisliu/datasets/scannet/images/scans/scene0000_00 \
  --processed-scene /home/lewisliu/datasets/scannet/processed/baseline30/openset_scans/scene0000_00 \
  --output /home/lewisliu/datasets/scannet-sg-processed/partaware_v1/scannet/scene0000_00/parts_new \
  --image-size 480 --limit 3 --visualize
```

Hypersim uses the preserved `scannet_sg_input` manifest interface:

```bash
.venv-vlpart/bin/python scannet/script/run_partaware.py \
  --manifest /home/lewisliu/datasets/scannet-sg-input/hypersim/ai_001_002/manifest.json \
  --processed-scene /home/lewisliu/datasets/scannet-sg-processed/hypersim_joint_v4/hypersim/ai_001_002 \
  --output /home/lewisliu/datasets/scannet-sg-processed/partaware_v1/hypersim/ai_001_002/parts_new \
  --image-size 480 --limit 3 --visualize
```

Remove `--limit 3` for all prepared frames. `--sam-checkpoint` optionally refines
boxes with the preserved SAM weight. `--dbscan-eps 0.05`, `--subtract-contained`,
and `--erode-pixels 1` are independent, disabled-by-default cleanup experiments.
Containment cleanup requires the same resolved parent instance, and ambiguous
ownership stays unresolved. `--part-overrides` accepts a JSON mapping from object
names to part names; objects absent from the knowledge base get no guessed parts.

Outputs include `partaware_graph.json`, independent per-frame Boolean masks,
observed part point clouds, `run_config.json`, and Chinese `run_zh.jsonl` logs.
The original `topology_map.json` is preserved. Parts have their own 1024-dimensional
CLIP RN50 feature space; confirmed, attached tracks create `part_of` edges.
See [interfaces](docs/PARTAWARE_INTERFACES.md) and the
[consolidated Chinese report](RESEARCH_LOG.md).

```bash
.venv-vlpart/bin/python -m unittest discover -s scannet/script/tests_partaware -v
```

Real ScanNet and Hypersim object regression runs produced identical graphs with
the saved binary and new default binary. Native VLPart and optional SAM runs
completed on small samples. These runs establish functionality and compatibility;
detector errors, duplicate parts, and uncertain ownership remain. Accuracy gains
require annotated evaluation. The consolidated report is in [RESEARCH_LOG.md](RESEARCH_LOG.md); raw records and generated graphs are stored under `/home/lewisliu/datasets/scannet-sg-processed/partaware_v1`.


### Visualize the generated part graph

The main experiment's `topology_map.json` is the saved object baseline. New parts
are in `parts/partaware_graph.json` or `parts_refined/partaware_graph.json`.
Re-fused object graphs are under the external experiment's `cpp_regression`.

```bash
.venv-vlpart/bin/python script/visualize_partaware.py \
  --graph /home/lewisliu/datasets/scannet-sg-processed/partaware_v1/hypersim/ai_001_002/parts_refined/partaware_graph.json \
  --base-cloud /home/lewisliu/datasets/scannet-sg-processed/hypersim_joint_v4/hypersim/ai_001_002/instance_cloud_with_background.ply \
  --show-object-edges
```

Blue spheres represent objects; colored part clouds and spheres show confirmed
tracks; green lines are `part_of`. The terminal lists node IDs and names.
`--include-provisional` displays all tracks; `--nodes-only` hides part point clouds;
`--check-only` validates geometry without opening a window. The original
`script/visualize_map.py` remains available for object graphs.

## Dataset Download
To download our dataset, please check [here](/download/Download_ScanNet_SG.md)


## Environment Installation

This section explains how to prepare your machine to work with ScanNet-SG. What you install depends on how you plan to use the project: many users only need 1) a lightweight setup to load the data and run the Python utilities, while others will 2) reproduce our full pipeline for building new scene graphs and alignment data. The instructions below walk through both cases step by step. For 1), we provide both python and C++ usage interface and examples.

Clone code:
```bash
git clone git@github.com:tud-amr/ScanNet-SG.git --recurse-submodule
cd ScanNet-SG
```

__1) Usage only environment__

If you only want to use the dataset, install the environment as follows (using mamba instead of conda will be much faster):
```bash
conda create -n scannet-sg python=3.10
conda activate scannet-sg
conda install -c conda-forge numpy matplotlib -y
# If you wish to have the full visualization functions (for images in ScanNet), also install opencv with the following command
pip install opencv-python open3d
```

To use C++ interface, do the following:
```bash
cmake -S src -B build_read_and_visualize_map
cmake --build build_read_and_visualize_map
```

__2) Environment for building new scene graphs and alignment data__

If you wish to generate new scene graphs and alignment data with our tools, install the environment by:

```bash
conda create -n scannet-sg python=3.10
conda activate scannet-sg
```

The following environment installation basically follows the requirements of groundedSAM. (groundedSAM code requires python>=3.8, as well as pytorch>=1.7 and torchvision>=0.8):

```bash
cd scannet/script/thirdparty/Grounded-Segment-Anything
export AM_I_DOCKER=False
export BUILD_WITH_CUDA=True


# Install PyTorch first (GroundingDINO editable install imports torch at build time).
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=11.8

python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade diffusers[torch]

git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh

cd ..
git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

pip install -v ipython ipykernel
pip install -v onnx onnxruntime
pip install -v matplotlib opencv-python pycocotools
pip install -v open3d
```

Then download pretrained weights of GroundedSeg and SAM
```
cd Grounded-Segment-Anything
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

Install sentence transformers (pin versions to avoid `recognize-anything` / RAM++ and `sentence-transformers` / `transformers` incompatibilities):
```
pip uninstall -y transformers tokenizers sentence-transformers
pip install "transformers==4.35.2" "tokenizers==0.14.1"
pip install "sentence-transformers>=2.2.0,<3"
```


__Some generation scripts also call C++ tools__ (for example `openset_ply_map` and `generate_json`). 
To compile these tools, please check [Build the C++ tools](scannet/readme_openset.md#build-the-c-tools-cmake)


## Map Interface Usage

### Python version:
The Python class for I/O of the SceneGraph (or TopologyMap) JSON file is defined in `script/include/topology_map.py` as `TopologyMap`.
Check the examples below to learn how to use the interface. (The interface also contains free-space nodes, but we do not use them in the current version.)

- Read a scene graph from a json file
```bash
python script/read_map.py
```

- Visualize a scene graph using the following command
```bash
python script/visualize_map.py --show_bboxes --show_edges
```
Add `--map_ply_path xxx.ply --topology_map_path xxx.json` to specify the data. Add `--enable_picking` to use interactive mode: the name of a node will be printed when you press `Shift` and left-click the blue sphere of a node.
By default, example data in `sample_data/scans/scene0000_00` will be used. You will see an image like the following:

![image](/sample_data/scans/scene0000_00/scene_0000.png)



- Generate a random scene graph
```bash
python script/random_map_generator.py
```

### C++ version:

C++ data structure is defined in `include/topology_map.h` 
Check the example in the following to know how to use the C++ version interface.

- Read and visualize a scene graph
```bash
./read_and_visualize_map <map_file>
```


## Generate Scene Graphs with Your Own Data
Please refer to [OpenSet F2S data generation](scannet/readme_openset.md) and [S2S data generation](scannet/readme_subscan.md)


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
