import hashlib,json,struct,time,zlib
from pathlib import Path
import cv2,numpy as np,requests

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda : file.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()

def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        return
    partial = dest.with_suffix(dest.suffix + '.part')
    if dest.suffix in ('.txt', '.json'):
        session = requests.Session()
        session.proxies.update(http='http://127.0.0.1:17890', https='http://127.0.0.1:17890')
        response = session.get(url, headers={'Accept-Encoding': 'identity'}, timeout=(25, 60))
        response.raise_for_status()
        body = response.content
        if dest.suffix == '.json':
            json.loads(body)
        else:
            assert b'=' in body and (not body.lstrip().startswith(b'<'))
        partial.write_bytes(body)
        partial.replace(dest)
        session.close()
        print('METADATA', dest.name, len(body), flush=True)
        return
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        session = requests.Session()
        session.proxies.update(http='http://127.0.0.1:17890', https='http://127.0.0.1:17890')
        try:
            response = session.get(url, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'}, timeout=(25, 60), stream=True)
            response.raise_for_status()
            content_range = response.headers.get('Content-Range')
            if response.status_code == 206:
                assert content_range and int(content_range.split(' ')[1].split('-')[0]) == offset
                total = int(content_range.rsplit('/', 1)[1])
            else:
                assert offset == 0, 'Server ignored resume range'
                total = int(response.headers['Content-Length'])
            print('DOWNLOAD', dest.name, 'total', total, 'offset', offset, flush=True)
            with partial.open('ab') as output:
                for block in response.iter_content(1024 ** 2):
                    if block:
                        output.write(block)
            assert partial.stat().st_size == total, 'Incomplete transfer'
            partial.replace(dest)
            response.close()
            session.close()
            return
        except (requests.RequestException, AssertionError) as error:
            print('RETRY', dest.name, attempt + 1, type(error).__name__, flush=True)
            session.close()
            if attempt == 4:
                raise
            time.sleep(2)

def export(sens, dest, scene):
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / 'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        assert old['sampling']['frame_step'] == 40
        return old
    with sens.open('rb') as source:

        def unpack(fmt):
            return struct.unpack('<' + fmt, source.read(struct.calcsize('<' + fmt)))
        assert unpack('I')[0] == 4
        sensor = source.read(unpack('Q')[0]).decode('utf-8')
        kc = np.array(unpack('16f')).reshape(4, 4)
        ec = np.array(unpack('16f')).reshape(4, 4)
        kd = np.array(unpack('16f')).reshape(4, 4)
        ed = np.array(unpack('16f')).reshape(4, 4)
        (cc, dc) = unpack('2i')
        assert cc == 2 and dc == 1
        (cw, ch, dw, dh) = unpack('4I')
        scale = unpack('f')[0]
        total = unpack('Q')[0]
        records = []
        invalid = []
        for index in range(total):
            pose = np.array(unpack('16f')).reshape(4, 4)
            (tc, td, cs, ds) = unpack('4Q')
            if index % 40:
                source.seek(cs + ds, 1)
                continue
            color = source.read(cs)
            depth = source.read(ds)
            if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1], atol=0.0001):
                invalid.append(index)
                continue
            frame = f'{index:06d}'
            color_path = dest / f'frame-{frame}.color.jpg'
            depth_path = dest / f'frame-{frame}.depth.png'
            pose_path = dest / f'frame-{frame}.pose.txt'
            color_path.write_bytes(color)
            values = np.frombuffer(zlib.decompress(depth), dtype='<u2').reshape(dh, dw)
            assert cv2.imwrite(str(depth_path), values)
            np.savetxt(pose_path, pose, fmt='%.9g')
            records.append(dict(frame_id=frame, source_frame_id=index, rgb=color_path.name, depth=depth_path.name, pose=pose_path.name))
    fields = dict(m_colorWidth=cw, m_colorHeight=ch, m_depthWidth=dw, m_depthHeight=dh, m_depthShift=scale, m_calibrationColorIntrinsic=' '.join(map(str, kc.ravel())), m_calibrationDepthIntrinsic=' '.join(map(str, kd.ravel())), m_calibrationColorExtrinsic=' '.join(map(str, ec.ravel())), m_calibrationDepthExtrinsic=' '.join(map(str, ed.ravel())))
    (dest / '_info.txt').write_text('\n'.join((f'{key} = {value}' for (key, value) in fields.items())) + '\n')
    metadata = dict(format='scannet_sg_input', schema_version=1, dataset='scannet', scene_id=scene, camera_info='_info.txt', length_unit='meter', pose_convention='T_world_from_camera', world_frame='scannet_unaligned_world', depth_type='optical_axis_z', depth_encoding='uint16', depth_scale=float(scale), invalid_depth=0, source_sensor=sensor, source_sens=str(sens), sampling=dict(frame_step=40, frame_cap=None, original_frames=total, invalid_pose_frames=invalid), frames=records)
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print('EXPORTED', scene, len(records), 'of', total, 'frames', flush=True)
    return metadata
