#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path

import cv2


def hardlink_or_skip(src: Path, dst: Path):
    """Create a hardlink without duplicating data."""
    if dst.exists():
        return

    if not src.exists():
        raise FileNotFoundError(f"Missing source file: {src}")

    os.link(src, dst)


def parse_scannet_metadata(txt_path: Path):
    """
    Parse ScanNet's sceneXXXX_XX.txt metadata.
    """
    if not txt_path.exists():
        raise FileNotFoundError(
            f"ScanNet metadata file not found:\n{txt_path}"
        )

    metadata = {}

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            metadata[key.strip()] = value.strip()

    required = [
        "colorHeight",
        "colorWidth",
        "depthHeight",
        "depthWidth",
        "fx_color",
        "fy_color",
        "mx_color",
        "my_color",
        "fx_depth",
        "fy_depth",
        "mx_depth",
        "my_depth",
    ]

    missing = [k for k in required if k not in metadata]

    if missing:
        raise RuntimeError(
            "Missing required metadata fields: "
            + ", ".join(missing)
        )

    return metadata


def write_info_file(metadata, output_path: Path):
    """
    Generate the _info.txt format expected by ScanNet-SG C++.
    """

    cw = int(metadata["colorWidth"])
    ch = int(metadata["colorHeight"])

    dw = int(metadata["depthWidth"])
    dh = int(metadata["depthHeight"])

    fx_c = float(metadata["fx_color"])
    fy_c = float(metadata["fy_color"])
    cx_c = float(metadata["mx_color"])
    cy_c = float(metadata["my_color"])

    fx_d = float(metadata["fx_depth"])
    fy_d = float(metadata["fy_depth"])
    cx_d = float(metadata["mx_depth"])
    cy_d = float(metadata["my_depth"])

    # ScanNet depth values are stored in millimeters.
    depth_shift = 1000

    text = f"""m_depthWidth = {dw}
m_depthHeight = {dh}
m_colorWidth = {cw}
m_colorHeight = {ch}
m_depthShift = {depth_shift}
m_calibrationColorIntrinsic = {fx_c} 0 {cx_c} 0 0 {fy_c} {cy_c} 0 0 0 1 0 0 0 0 1
m_calibrationDepthIntrinsic = {fx_d} 0 {cx_d} 0 0 {fy_d} {cy_d} 0 0 0 1 0 0 0 0 1
"""

    output_path.write_text(text, encoding="utf-8")


def get_frame_ids(exported_dir: Path):
    """
    Get frame IDs from exported color images.
    Expected:
        color/0.jpg
        color/1.jpg
        ...
    """

    color_dir = exported_dir / "color"

    if not color_dir.exists():
        raise FileNotFoundError(
            f"Color directory not found: {color_dir}"
        )

    frame_ids = []

    for path in color_dir.glob("*.jpg"):
        if re.fullmatch(r"\d+", path.stem):
            frame_ids.append(int(path.stem))

    frame_ids.sort()

    if not frame_ids:
        raise RuntimeError(
            f"No exported RGB frames found in {color_dir}"
        )

    return frame_ids


def prepare_scene(scene_name, raw_root, output_root):
    raw_scene = raw_root / scene_name

    if not raw_scene.exists():
        raise FileNotFoundError(
            f"Raw ScanNet scene does not exist:\n{raw_scene}"
        )

    exported = raw_scene / "exported"

    if not exported.exists():
        raise FileNotFoundError(
            f"""
Exported ScanNet data not found:

{exported}

Export the .sens file first with ScanNet SensReader.
Expected directories:

exported/color/
exported/depth/
exported/pose/
exported/intrinsic/
"""
        )

    output_scene = output_root / scene_name
    output_scene.mkdir(parents=True, exist_ok=True)

    print(f"Scene:        {scene_name}")
    print(f"Raw scene:    {raw_scene}")
    print(f"Exported:     {exported}")
    print(f"Output scene: {output_scene}")
    print()

    # ---------------------------------------------------------
    # 1. Parse original ScanNet metadata
    # ---------------------------------------------------------

    metadata_txt = raw_scene / f"{scene_name}.txt"

    print("[1/5] Reading ScanNet metadata")

    metadata = parse_scannet_metadata(metadata_txt)

    print(
        f"      RGB:   "
        f"{metadata['colorWidth']} x {metadata['colorHeight']}"
    )

    print(
        f"      Depth: "
        f"{metadata['depthWidth']} x {metadata['depthHeight']}"
    )

    # ---------------------------------------------------------
    # 2. Generate _info.txt
    # ---------------------------------------------------------

    print("[2/5] Generating _info.txt")

    info_path = output_scene / "_info.txt"
    write_info_file(metadata, info_path)

    # ---------------------------------------------------------
    # 3. Detect exported frames
    # ---------------------------------------------------------

    print("[3/5] Detecting frames")

    frame_ids = get_frame_ids(exported)

    print(f"      Found {len(frame_ids)} RGB frames")

    # ---------------------------------------------------------
    # 4. Create RGB / pose hardlinks
    # ---------------------------------------------------------

    print("[4/5] Creating RGB and pose hardlinks")

    rgb_count = 0
    pose_count = 0

    for frame_id in frame_ids:

        n = f"{frame_id:06d}"

        rgb_src = exported / "color" / f"{frame_id}.jpg"
        pose_src = exported / "pose" / f"{frame_id}.txt"

        rgb_dst = output_scene / f"frame-{n}.color.jpg"
        pose_dst = output_scene / f"frame-{n}.pose.txt"

        if rgb_src.exists():
            hardlink_or_skip(rgb_src, rgb_dst)
            rgb_count += 1

        if pose_src.exists():
            hardlink_or_skip(pose_src, pose_dst)
            pose_count += 1

    print(f"      RGB:  {rgb_count}")
    print(f"      Pose: {pose_count}")

    # ---------------------------------------------------------
    # 5. Convert depth PNG -> uint16 PGM
    # ---------------------------------------------------------

    print("[5/5] Preparing depth images")

    depth_count = 0

    expected_shape = (
        int(metadata["depthHeight"]),
        int(metadata["depthWidth"]),
    )

    for index, frame_id in enumerate(frame_ids, start=1):

        n = f"{frame_id:06d}"

        src = exported / "depth" / f"{frame_id}.png"
        dst = output_scene / f"frame-{n}.depth.pgm"

        if dst.exists():
            depth_count += 1
            continue

        if not src.exists():
            print(f"      WARNING: missing depth {src}")
            continue

        depth = cv2.imread(
            str(src),
            cv2.IMREAD_UNCHANGED
        )

        if depth is None:
            raise RuntimeError(
                f"Failed to read depth image: {src}"
            )

        if depth.dtype.name != "uint16":
            raise RuntimeError(
                f"Unexpected depth dtype: "
                f"{src} -> {depth.dtype}"
            )

        if depth.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected depth resolution: "
                f"{src} -> {depth.shape}, "
                f"expected {expected_shape}"
            )

        ok = cv2.imwrite(str(dst), depth)

        if not ok:
            raise RuntimeError(
                f"Failed to write PGM: {dst}"
            )

        depth_count += 1

        if index % 500 == 0:
            print(
                f"      Processed "
                f"{index}/{len(frame_ids)} frames"
            )

    print()
    print("======================================")
    print("ScanNet-SG scene preparation SUCCESS")
    print("======================================")
    print(f"Scene:       {scene_name}")
    print(f"RGB frames:  {rgb_count}")
    print(f"Depth:       {depth_count}")
    print(f"Poses:       {pose_count}")
    print(f"Output:      {output_scene}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a ScanNet scene for ScanNet-SG"
    )

    parser.add_argument(
        "scene",
        help="Scene name, e.g. scene0000_00"
    )

    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path.home()
        / "datasets"
        / "scannet"
        / "scans",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home()
        / "datasets"
        / "scannet"
        / "images"
        / "scans",
    )

    args = parser.parse_args()

    prepare_scene(
        args.scene,
        args.raw_root.expanduser(),
        args.output_root.expanduser(),
    )


if __name__ == "__main__":
    main()
