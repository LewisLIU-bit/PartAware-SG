"""Download selected official Hypersim RGB-D frames and independent labels."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import zlib
import zipfile

import cv2
import h5py
import numpy as np
import requests

ARCHIVE_ROOT = 'https://docs-assets.developer.apple.com/ml-research/datasets/hypersim/v1/scenes/'
CAMERA_METADATA = ('https://raw.githubusercontent.com/apple/ml-hypersim/main/'
                   'contrib/mikeroberts3000/metadata_camera_parameters.csv')


def request(url, **kwargs):
    for attempt in range(4):
        try:
            result = requests.get(url, timeout=(15, 90), **kwargs)
            result.raise_for_status()
            return result
        except requests.RequestException:
            if attempt == 3:
                raise


class RemoteFile(io.RawIOBase):
    """Seekable HTTP range reader used only to inspect a ZIP central directory."""
    def __init__(self, url):
        self.url = url
        response = requests.head(url, timeout=(15, 45))
        response.raise_for_status()
        self.length = int(response.headers['Content-Length'])
        self.position = 0

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else (self.position if whence == 1 else self.length) + offset
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b''
        data = range_bytes(self.url, self.position, size)
        self.position += len(data)
        return data

    def seekable(self):
        return True


def range_bytes(url, offset, size):
    response = request(url, headers={'Range': f'bytes={offset}-{offset + size - 1}'})
    if response.status_code != 206 or len(response.content) != size:
        raise RuntimeError('The server did not honor the exact archive range')
    return response.content


def extract_member(url, info, destination):
    if destination.exists():
        data = destination.read_bytes()
        if len(data) == info.file_size and zlib.crc32(data) & 0xffffffff == info.CRC:
            return len(data), False
        raise RuntimeError(f'Existing source has a different checksum: {destination}')
    header = range_bytes(url, info.header_offset, 30)
    values = struct.unpack('<4s5H3I2H', header)
    if values[0] != b'PK\x03\x04' or values[2] & 1:
        raise ValueError('Unsupported encrypted or malformed ZIP member')
    offset = info.header_offset + 30 + values[-2] + values[-1]
    encoded = range_bytes(url, offset, info.compress_size)
    if info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(encoded, -15)
    elif info.compress_type == zipfile.ZIP_STORED:
        data = encoded
    else:
        raise ValueError('Unsupported ZIP compression')
    if len(data) != info.file_size or zlib.crc32(data) & 0xffffffff != info.CRC:
        raise RuntimeError(f'CRC verification failed: {info.filename}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    return len(data), True


def hdf(path):
    with h5py.File(path, 'r') as file:
        return file['dataset'][:]


def prepare(args):
    if not re.fullmatch(r'ai_\d{3}_\d{3}', args.scene):
        raise ValueError('Invalid official Hypersim scene name')
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    native = output / 'source_hdf5'
    url = ARCHIVE_ROOT + args.scene + '.zip'
    print('正在核查官方场景压缩包目录', args.scene, flush=True)
    with zipfile.ZipFile(RemoteFile(url)) as archive:
        members = {info.filename: info for info in archive.infolist()}
    prefix = args.scene + '/'
    existing = None
    if args.existing_manifest:
        original = Path(args.existing_manifest).expanduser().resolve()
        existing = json.loads(original.read_text())
        if existing['scene_id'] != args.scene:
            raise ValueError('Existing manifest scene disagrees with the requested archive')
        excluded = []
        valid_frames = []
        for frame in existing['frames']:
            rgb = cv2.imread(str(original.parent / frame['rgb']))
            depth = cv2.imread(str(original.parent / frame['depth']), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
                excluded.append({'frame_id': frame['frame_id'], 'reason': 'RGB and registered depth dimensions disagree'})
            else:
                valid_frames.append(frame)
        existing['frames'] = valid_frames
        existing['excluded_input_frames'] = excluded
        print('既有输入验证完成，保留', len(valid_frames), '帧，排除不匹配帧', excluded, flush=True)
        selected = [(f['source_camera'], int(f['source_camera_frame'])) for f in existing['frames']]
        candidates = selected
    else:
        pattern = re.compile(re.escape(prefix) + r'images/scene_(cam_\d+)_geometry_hdf5/frame\.(\d+)\.depth_meters\.hdf5$')
        available = sorted((m.group(1), int(m.group(2))) for name in members if (m := pattern.fullmatch(name))
                           and all(prefix + path in members for path in [
                               f'images/scene_{m.group(1)}_final_hdf5/frame.{int(m.group(2)):04d}.color.hdf5',
                               f'images/scene_{m.group(1)}_geometry_hdf5/frame.{int(m.group(2)):04d}.semantic.hdf5',
                               f'images/scene_{m.group(1)}_geometry_hdf5/frame.{int(m.group(2)):04d}.semantic_instance.hdf5']))
        if not available:
            raise ValueError('No complete official frames are available')
        selected = available[::args.frame_step][:args.max_frames]
        # Retain the candidate source interval, then sample; the cap applies to selected frames.
        candidates = available[:min(len(available), args.max_frames * args.frame_step)]
        print('官方场景总帧数', len(available), '三选一后入选', len(selected), flush=True)
    needed = set()
    for camera, frame in candidates:
        stem = f'frame.{frame:04d}'
        for suffix in ['semantic.hdf5', 'semantic_instance.hdf5']:
            needed.add(prefix + f'images/scene_{camera}_geometry_hdf5/{stem}.{suffix}')
        if not existing:
            needed.add(prefix + f'images/scene_{camera}_geometry_hdf5/{stem}.depth_meters.hdf5')
            needed.add(prefix + f'images/scene_{camera}_final_hdf5/{stem}.color.hdf5')
    for camera in {c for c, _ in candidates}:
        for suffix in ['positions', 'orientations', 'frame_indices']:
            name = prefix + f'_detail/{camera}/camera_keyframe_{suffix}.hdf5'
            if name in members:
                needed.add(name)
    needed.add(prefix + '_detail/metadata_scene.csv')
    needed.update(name for name in members if name.startswith(prefix + '_detail/mesh/')
                  and ('semantic_instance_bounding_box' in name or name.endswith(('mesh_objects_si.hdf5', 'mesh_objects_sii.hdf5'))))
    missing = needed.difference(members)
    if missing:
        raise FileNotFoundError(f'Missing official source files: {sorted(missing)[:3]}')
    completed, byte_count = 0, 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(extract_member, url, members[name], native / name[len(prefix):]): name
                   for name in sorted(needed)}
        for future in as_completed(futures):
            size, downloaded = future.result()
            byte_count += size
            completed += 1
            if completed % 20 == 0 or completed == len(futures):
                print(f'官方文件校验完成 {completed}/{len(futures)}，累计 {byte_count / 1e6:.1f} MB', flush=True)
    metadata_path = native / 'metadata_camera_parameters.csv'
    metadata_bytes = metadata_path.read_bytes() if metadata_path.exists() else request(CAMERA_METADATA).content
    metadata_path.write_bytes(metadata_bytes)
    camera_row = next(r for r in csv.DictReader(io.StringIO(metadata_bytes.decode())) if r['scene_name'] == args.scene)
    width, height = int(float(camera_row['settings_output_img_width'])), int(float(camera_row['settings_output_img_height']))
    projection = np.array([float(camera_row[f'M_proj_{i}{j}']) for i in range(4) for j in range(4)]).reshape(4, 4)
    if abs(projection[0, 1]) + abs(projection[1, 0]) > 1e-8:
        raise ValueError('Skewed cameras cannot use the existing pinhole interface')
    fx, fy = projection[0, 0] * width / 2, projection[1, 1] * height / 2
    cx, cy = (width * (1 - projection[0, 2]) - 1) / 2, (height * (1 + projection[1, 2]) - 1) / 2
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    rays = np.array([float(camera_row[f'M_cam_from_uv_{i}{j}']) for i in range(3) for j in range(3)]).reshape(3, 3)
    uu, vv = np.meshgrid(np.linspace(-1 + 1 / width, 1 - 1 / width, width), np.linspace(1 - 1 / height, -1 + 1 / height, height))
    directions = np.stack([uu, vv, np.ones_like(uu)], axis=-1) @ rays.T
    z_ratio = np.abs(directions[..., 2]) / np.linalg.norm(directions, axis=-1)
    scene_rows = list(csv.DictReader((native / '_detail/metadata_scene.csv').read_text().splitlines()))
    if 'meters_per_asset_unit' in scene_rows[0]:
        scale = float(scene_rows[0]['meters_per_asset_unit'])
    else:
        scale = float(next(row['parameter_value'] for row in scene_rows
                           if row['parameter_name'] == 'meters_per_asset_unit'))
    poses = {}
    for camera in {c for c, _ in selected}:
        positions = hdf(native / f'_detail/{camera}/camera_keyframe_positions.hdf5')
        rotations = hdf(native / f'_detail/{camera}/camera_keyframe_orientations.hdf5')
        indices_path = native / f'_detail/{camera}/camera_keyframe_frame_indices.hdf5'
        indices = hdf(indices_path).reshape(-1).astype(int) if indices_path.exists() else np.arange(len(positions))
        poses[camera] = {int(index): (position, rotation) for index, position, rotation in zip(indices, positions, rotations)}
    if existing:
        manifest = existing
        # Existing RGB-D is reused byte-for-byte; official labels are kept independent.
        for frame in manifest['frames']:
            for field in ['rgb', 'depth', 'pose']:
                frame[field] = str((original.parent / frame[field]).resolve())
        manifest['camera_info'] = str((original.parent / manifest['camera_info']).resolve())
    else:
        matrix = np.eye(4)
        matrix[:3, :3] = k
        (output / '_info.txt').write_text('\n'.join([
            f'm_colorWidth = {width}', f'm_colorHeight = {height}',
            f'm_depthWidth = {width}', f'm_depthHeight = {height}', 'm_depthShift = 1000',
            'm_calibrationColorIntrinsic = ' + ' '.join(map(str, matrix.reshape(-1))),
            'm_calibrationDepthIntrinsic = ' + ' '.join(map(str, matrix.reshape(-1)))]) + '\n')
        manifest = {'schema_version': 1, 'format': 'scannet_sg_input', 'dataset': 'hypersim',
                    'scene_id': args.scene, 'camera_info': '_info.txt', 'length_unit': 'meter',
                    'pose_convention': 'T_world_from_camera', 'world_frame': 'hypersim_world_z_up',
                    'depth_type': 'optical_axis_z', 'depth_scale': 1000.0, 'frames': []}
        for camera, index in selected:
            fid = f'{camera}_{index:04d}'
            color = hdf(native / f'images/scene_{camera}_final_hdf5/frame.{index:04d}.color.hdf5')
            brightness = color @ np.array([0.3, 0.59, 0.11])
            percentile = np.percentile(brightness[np.isfinite(brightness) & (brightness > 0)], 90)
            rgb = np.clip(np.maximum(color * (0.8 ** 2.2 / max(percentile, 1e-8)), 0) ** (1 / 2.2), 0, 1)
            rgb = np.round(rgb * 255).astype(np.uint8)
            distance = hdf(native / f'images/scene_{camera}_geometry_hdf5/frame.{index:04d}.depth_meters.hdf5')
            z = distance * z_ratio
            valid = np.isfinite(z) & (z > 0) & (z < 65.535)
            depth = np.where(valid, np.round(np.nan_to_num(z) * 1000), 0).astype(np.uint16)
            position, rotation = poses[camera][index]
            pose = np.eye(4)
            pose[:3, :3] = rotation @ np.diag([1, -1, -1])
            pose[:3, 3] = position * scale
            if not cv2.imwrite(str(output / f'{fid}.color.png'), rgb[..., ::-1]):
                raise RuntimeError('RGB encoding failed')
            if not cv2.imwrite(str(output / f'{fid}.depth.pgm'), depth):
                raise RuntimeError('Depth encoding failed')
            np.savetxt(output / f'{fid}.pose.txt', pose, fmt='%.10f')
            manifest['frames'].append({'frame_id': fid, 'source_camera': camera, 'source_camera_frame': index,
                                      'rgb': f'{fid}.color.png', 'depth': f'{fid}.depth.pgm', 'pose': f'{fid}.pose.txt'})
    manifest['ground_truth_source'] = str(native)
    manifest['sampling'] = {'candidate_frames': len(candidates), 'frame_step': args.frame_step if not existing else 1,
                            'selected_frames': len(selected), 'candidate_ids': [f'{c}_{i:04d}' for c, i in candidates]}
    manifest['sampling']['total_available_frames'] = len(available) if not existing else len(selected)
    manifest['sampling']['maximum_selected_frames'] = args.max_frames
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    record = {'archive_url': url, 'candidate_frames': len(candidates), 'selected_frames': len(selected),
              'official_files': len(needed), 'source_uncompressed_bytes': byte_count,
              'meters_per_asset_unit': scale, 'intrinsics': k.tolist(),
              'camera_metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
              'reused_rgbd_manifest': str(original) if existing else None}
    (output / 'input_record.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    print('输入准备完成', args.scene, len(selected), '帧', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-frames', type=int, default=150)
    parser.add_argument('--frame-step', type=int, default=3)
    parser.add_argument('--existing-manifest')
    options = parser.parse_args()
    if options.max_frames < 1 or options.frame_step < 1:
        parser.error('Sampling counts must be positive')
    prepare(options)
