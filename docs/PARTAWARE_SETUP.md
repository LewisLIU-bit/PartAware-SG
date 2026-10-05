# Reproducing the project inference environments

The current WSL project is already installed and tested. Use the existing
`scannet-sg` conda environment for the original object frontend and
`.venv-vlpart/bin/python` for the part frontend. This separation preserves the
original timm dependency. Dataset preparation uses the commands in the main
README. Original baseline model installation follows the
[upstream ScanNet-SG README](https://github.com/tud-amr/ScanNet-SG#environment-installation).

## Fresh installation on the verified machine

These commands assume a working ScanNet-SG environment, CUDA toolkit, compiler,
CLIP package, SciPy, scikit-learn, and OpenCV. They do not install or replace
PyTorch. The verified GPU is an RTX 5070 Laptop with CUDA 13; adjust CUDA paths
and `TORCH_CUDA_ARCH_LIST` for other machines.

```bash
cd /home/lewisliu/PartAware-SG
BASE_PYTHON=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
"$BASE_PYTHON" -m venv --system-site-packages .venv-vlpart
.venv-vlpart/bin/python -m pip install -r docs/partaware_requirements.txt
.venv-vlpart/bin/python -m pip install --no-deps timm==0.5.4

git clone https://github.com/facebookresearch/VLPart.git scannet/script/thirdparty/VLPart
git -C scannet/script/thirdparty/VLPart checkout e48903ab0cb38e4dd5be7ce7da2e36c855eff53f
git clone https://github.com/facebookresearch/detectron2.git scannet/script/thirdparty/detectron2
git -C scannet/script/thirdparty/detectron2 checkout 1e3e13bbf607b54f62205c4c33922521822fb298

CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  .venv-vlpart/bin/python -m pip install --no-build-isolation --no-deps \
  -e scannet/script/thirdparty/detectron2
```

Run clone commands only when their target directories do not exist. Existing
source trees and weights in this project are intentionally ignored by Git;
`sources.lock.json` records their revisions and `model_checksums.json` records
downloaded files. OP3DSG knowledge is vendored in `scannet/script/partaware/object_part_knowledge.json`
with a source provenance notice. No OP3DSG checkout, training, or LLM service is required by this adapter.

## Official inference weights

```bash
mkdir -p checkpoints/vlpart checkpoints/clip
curl -fL --retry 3 \
  https://github.com/PeizeSun/VLPart/releases/download/v0.1/swinbase_cascade_lvis_paco.pth \
  -o checkpoints/vlpart/swinbase_cascade_lvis_paco.pth
curl -fL --retry 3 \
  https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt \
  -o checkpoints/clip/RN50.pt
sha256sum checkpoints/vlpart/swinbase_cascade_lvis_paco.pth checkpoints/clip/RN50.pt
```

Expected SHA-256 values:

- VLPart: `e524eb802224b0ac9069826122c7423fc612c436b4d28d76fad19dca0fd1d798`.
- RN50: `afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762`.

The optional SAM switch uses the already preserved
`scannet/script/thirdparty/Grounded-Segment-Anything/sam_vit_h_4b8939.pth`.

## Checks

V3 adds only the official HDF5 reader to the existing object environment:

```bash
conda activate scannet-sg
python -m pip install --no-deps h5py==3.11.0
```

The independent evaluator uses the already installed OpenAI `clip` package and
official `ViT-B/16` weights, cached in `checkpoints/clip/ViT-B-16.pt`. These weights
are used for label retrieval, separately from VLPart's RN50 masked image features.
No global Python installation is modified. GPU model stages share an environment
lock under `~/.cache/partaware-sg/gpu.lock` so separate scene runs do not load their
large models simultaneously.

```bash
python -m unittest discover \
  -s scannet/script/tests_partaware -v
.venv-vlpart/bin/python scannet/script/run_partaware.py --help
cmake -S scannet -B scannet/build-partaware
cmake --build scannet/build-partaware -j2
```

See the main README for actual ScanNet and Hypersim invocation examples. The
small validation runs use `--image-size 480`; the CLI default is 640. Run the
part branch with native VLPart masks first, then evaluate each optional SAM,
DBSCAN, containment, and erosion switch separately. Inference completion alone
does not demonstrate mapping accuracy.

## Baseline compatibility patch

The saved object frontend includes two upstream GroundingDINO compatibility
fixes. A fresh recursive checkout can apply them before compiling GroundingDINO:

```bash
git -C scannet/script/thirdparty/Grounded-Segment-Anything \
  apply --check ../grounded_sam_compat.patch
git -C scannet/script/thirdparty/Grounded-Segment-Anything \
  apply ../grounded_sam_compat.patch
```

Do not apply the patch twice. The current WSL runtime already contains it.
The patch preserves earlier CUDA/inference fixes without publishing modified
third-party repositories. Outputs belong under `datasets/scannet-sg-processed`
and experiment names should end in `_v1` for the first run.

## Default Florence frontend

Florence runs in a separate long-lived Python worker, using the existing
`sg-florence` environment on this machine. This leaves RAM/GroundingDINO's
transformers 4.35 and VLPart's timm 0.5 dependencies intact. No API is used.

```bash
export FLORENCE_PYTHON=/home/lewisliu/miniconda3/envs/sg-florence/bin/python
export FLORENCE_MODEL_DIR=/home/lewisliu/models/vision/Florence-2-large-ft
```

On a fresh machine, prepare an isolated environment with a working CUDA
PyTorch matching the base environment, `transformers==4.49.0`,
`timm==1.0.15`, `einops==0.8.1`, and Pillow. Download the official
[Microsoft Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft)
checkpoint into `FLORENCE_MODEL_DIR`. Eager attention is used; FlashAttention
is optional. The loader scopes its optional-import workaround to Florence's
conditional import and passes an explicit trusted local configuration.

`get_seg_openset.py` defaults to `--grounding_backend florence` for both
`--image_folder` and `--manifest`. `--florence_model_dir` overrides the model
location. External proposal and saved rendering modes keep their own path.
Set `--grounding_backend dino` to reproduce the original frontend.
Use a fresh output folder for new inference; existing complete runs can still
be skipped with `--skip_existing`. Model and input failures are reported,
and never silently converted to a DINO run.


## V4 YOLOE and AdaPoinTr environments

These isolated environments inherit the existing base packages without upgrading
PyTorch, transformers or the original part runtime. The verified base is Python
3.10, PyTorch 2.14/CUDA 13 and approximately 8 GB GPU memory. Install only these
pinned additions; the GPU lock serializes large inference models.

```bash
BASE_PYTHON=/home/lewisliu/miniconda3/envs/scannet-sg/bin/python
"$BASE_PYTHON" -m venv --system-site-packages .venv-yoloe
.venv-yoloe/bin/python -m pip install --no-deps ultralytics==8.3.221 ultralytics-thop==2.2.2
"$BASE_PYTHON" -m venv --system-site-packages .venv-completion
.venv-completion/bin/python -m pip install --no-deps easydict==1.13 einops==0.8.1
git clone https://github.com/yuxumin/PoinTr.git scannet/script/thirdparty/PoinTr
git -C scannet/script/thirdparty/PoinTr checkout 4603257ed3db9e7dad349b712e1b2fe0da207015
mkdir -p checkpoints/yoloe checkpoints/completion
curl -fL --retry 3 https://huggingface.co/jameslahm/yoloe/resolve/main/yoloe-v8s-seg.pt \
  -o checkpoints/yoloe/yoloe-v8s-seg.pt
curl -fL --retry 3 https://github.com/ultralytics/assets/releases/download/v8.3.0/mobileclip_blt.ts \
  -o checkpoints/yoloe/mobileclip_blt.ts
curl -fL --retry 3 \
  'https://drive.usercontent.google.com/download?id=17pE2U2T2k4w1KfmDbL6U-GkEwD-duTaF&export=download&confirm=t' \
  -o checkpoints/completion/AdaPoinTr_PCN.pth
```

The author-provided **PCN** checkpoint is used, not Projected-ShapeNet55. It is
strictly loaded into the official AdaPoinTr architecture. The project adapter
provides portable PyTorch FPS, gathering and interpolation for inference, avoiding
historical CUDA extension builds. Its metric normalization and bounded alignment
adapt real partial observations; this is not a PCN benchmark reproduction or a
text-conditioned completion model. Verify hashes against `model_checksums.json`.

The source and weights remain local ignored runtime assets. YOLOE/Ultralytics
attribution and AGPL-3.0 provenance are recorded in `sources.lock.json`; AdaPoinTr's
source license remains in its checkout. Training-only dependencies are not installed.

## Offline mathematical report

`docs/RESEARCH_REPORT.md` is the single editable Chinese research source. General
sections describe the current version; individual version chapters contain history.
`docs/render_research_report.cjs` builds both the offline SVG-math HTML and LaTeX.
Use a Node.js >=20 runtime with `mathjax-full@3.2.2` and `marked@17.0.5` installed in
a project-local build environment, then pass its directory with `--dependencies`:

```bash
node docs/render_research_report.cjs --source docs/RESEARCH_REPORT.md \
  --output RESEARCH_LOG.html --latex-output RESEARCH_LOG.tex \
  --dependencies /path/to/report-build-environment
```

The published HTML requires no CDN, JavaScript, remote font or network connection.
