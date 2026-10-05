import argparse
import os
from tqdm import tqdm
from pathlib import Path
import sys


def _bootstrap_repo_root() -> None:
    cur = Path(__file__).resolve()
    for parent in [cur.parent, *cur.parents]:
        if (parent / "scannet" / "script" / "thirdparty").is_dir():
            p = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return


_bootstrap_repo_root()

from scannet.script.thirdparty.ensure_thirdparty import add_to_syspath, ensure_recognize_anything

# Lazily provision recognize-anything (provides the `ram` python package)
add_to_syspath(ensure_recognize_anything(from_file=__file__))

from inference_ram_plus_openset import RAMPlusOpensetInference

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tag2Text inferece for tagging and captioning')
    parser.add_argument('--scans_folder',
                        metavar='DIR',
                        help='path to imagescans folder',
                        default='/media/cc/My Passport/dataset/scannet/images/scans')
    parser.add_argument('--output_json_folder',
                        default='output_json',
                        help='path to output json folder')
    parser.add_argument('--start_scene_id',
                        default=200,
                        type=int,
                        help='start scene id (default: 200)')
    parser.add_argument('--end_scene_id',
                        default=400,
                        type=int,
                        help='end scene id (default: 400)')
    
    parser.add_argument('--image',
                        metavar='DIR',
                        help='path to dataset',
                        default='No need to be set')
    parser.add_argument('--pretrained',
                        metavar='DIR',
                        help='path to pretrained model',
                        default='/media/cc/Expansion/models/ram_plus_swin_large_14m.pth')
    parser.add_argument('--image_size',
                        default=384,
                        type=int,
                        metavar='N',
                        help='input image size (default: 448)')
    parser.add_argument('--similarity_threshold',
                        default=0.6,
                        type=float,
                        help='similarity threshold for filtering (default: 0.3)')
    parser.add_argument('--save_json',
                        default=True,
                        type=bool,
                        help='save json file (default: True)')

    parser.add_argument('--process_every_n_images',
                        default=3,
                        type=int,
                        help='process every n images (default: 3)')
    parser.add_argument('--llm_tag_des',
                        metavar='DIR',
                        help='path to LLM tag descriptions',
                        default='/home/cc/chg_ws/ros_ws/topomap_ws/src/semantic_topo_map/scannet/script/ram/scannet509.json')
    parser.add_argument('--native-vocabulary', action='store_true',
                        help='Use the RAM++ checkpoint vocabulary and calibrated thresholds')
    parser.add_argument("--manifest",
                        type=str,
                        default=None,
                        help="Input manifest; omit to use the legacy ScanNet folder mode",
    )
    parser.add_argument("--output_root",
                        type=str,
                        default=None,
                        help="Output root containing dataset and scene subdirectories",
    )

    args = parser.parse_args()
    if args.manifest is not None:
        if not args.output_root:
            parser.error("--output_root is required with --manifest")

        manifest_path = Path(args.manifest).expanduser().resolve()
        pretrained_path = Path(args.pretrained).expanduser().resolve()
        tag_description_path = None if args.native_vocabulary else Path(args.llm_tag_des).expanduser().resolve()

        for path in [manifest_path, pretrained_path] + ([tag_description_path] if tag_description_path else []):
            if not path.is_file():
                parser.error(f"Input file does not exist: {path}")

        print(f"Manifest: {manifest_path}")
        print(f"候选类别来源：{tag_description_path or 'RAM++ 原生词表'}")
        print("Processing all manifest frames without background filtering")

        ram_plus_openset_inference = RAMPlusOpensetInference(
            str(pretrained_path),
            args.image_size,
            str(tag_description_path) if tag_description_path else None,
        )

        ram_plus_openset_inference.run_manifest(
            manifest_path=manifest_path,
            output_root=args.output_root,
        )

        # Do not enter the legacy ScanNet folder loop.
        sys.exit(0)

    # get all the folders in the scans folder
    print("Loading scan folders...")
    scan_folders = [f for f in os.listdir(args.scans_folder) if os.path.isdir(os.path.join(args.scans_folder, f))]
    
    # filter the scans_folders by the start and end scene id
    print("Filtering scan folders...")
    filtered_scan_folders = []
    for scan_folder in scan_folders:
        scene_id = int(scan_folder.split('_')[0].split('scene')[-1])
        if scene_id >= args.start_scene_id and scene_id <= args.end_scene_id:
            filtered_scan_folders.append(scan_folder)

    print(f"Filtered scans folders: {filtered_scan_folders}")

    ram_plus_openset_inference = RAMPlusOpensetInference(
        args.pretrained, args.image_size, None if args.native_vocabulary else args.llm_tag_des)

    # Run the shared RAM++ frontend on each selected ScanNet folder.
    for scan_folder in tqdm(filtered_scan_folders):
        args.image = os.path.join(args.scans_folder, scan_folder)

        ram_plus_openset_inference.run_inference(args)
