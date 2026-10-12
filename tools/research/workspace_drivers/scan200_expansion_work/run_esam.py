"""Frozen author ESAM online inference on three registered ScanNet scans."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np,torch,open3d as o3d
WORK=Path(__file__).resolve().parent
BASE=Path('/home/lewisliu/baselines/ESAM')
import esam_cpu_pointops
import esam_cpu_compatibility
esam_cpu_compatibility.install()
sys.modules['pointops']=esam_cpu_pointops
sys.path.insert(0,'/home/lewisliu/baselines/FastSAM')
sys.path.insert(0,str(BASE))
from mmengine import Config
from mmengine.registry import init_default_scope
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample,PointData
import oneformer3d

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    adapter=esam_cpu_pointops.verify()
    order_adapter=esam_cpu_compatibility.verify()
    config_path=BASE/'configs/ESAM_CA/ESAM_online_scannet200_CA.py'
    config=Config.fromfile(str(config_path));init_default_scope('mmdet3d')
    model=MODELS.build(config.model).cpu().eval()
    weights=json.loads((WORK/'esam_weights.json').read_text())
    checkpoint=torch.load(weights['checkpoint'],map_location='cpu',weights_only=False)
    state=checkpoint['state_dict'];loaded=model.load_state_dict(state,strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    print('ESAM_OFFICIAL_WEIGHTS_LOADED_STRICT',len(state),'CPU',flush=True)
    (WORK/'esam_model_loaded.json').write_text(json.dumps(dict(checkpoint=weights,
        config_sha256=sha(config_path),state_dict_keys=len(state),strict_load=True,
        CPU_pointops=adapter,CPU_feature_order=order_adapter,training=False,test_cfg=dict(config.model.test_cfg)),indent=2)+'\n')
    if args.smoke:return
    registration=json.loads((WORK/'preregistration.json').read_text());rows=[]
    for scene in registration['scenes']:
        dest=BASE/'data/local_scan200'/scene
        while not (dest/'preparation.json').exists():time.sleep(5)
        prepared=json.loads((dest/'preparation.json').read_text())
        result=BASE/'work_dirs/local_scan200'/scene;result.mkdir(parents=True,exist_ok=True)
        marker=result/'inference.json'
        if marker.exists():
            previous=json.loads(marker.read_text())
            assert previous.get('CPU_feature_order')==order_adapter,'Refuse reuse of the invalid unordered CPU output'
            rows.append(previous);continue
        chunks=[np.load(r['path']) for r in prepared['frame_files']]
        points=np.stack([c['points'] for c in chunks]);sp=np.stack([c['superpoints'] for c in chunks])
        for chunk in chunks:chunk.close()
        mean=np.asarray(config.color_mean);std=np.asarray(config.color_std)
        points[:,:,3:]=(points[:,:,3:]-mean)/std
        inputs=dict(points=[torch.from_numpy(points.astype(np.float32))])
        sample=Det3DDataSample();sample.gt_pts_seg=PointData(sp_pts_mask=torch.from_numpy(sp))
        raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
        mesh_path=raw/(scene+'_vh_clean_2.ply')
        vertices=np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices).astype(np.float32)
        seg_path=raw/(scene+'_vh_clean_2.0.010000.segs.json')
        # Mesh segments are unsupervised reconstruction regions used by the
        # author's standard postprocess; no semantic/instance GT is given.
        segments=np.asarray(json.loads(seg_path.read_text())['segIndices'])
        sample.eval_ann_info=dict(rec_xyz=vertices,segment_ids=segments)
        start=time.monotonic();print('ESAM_INFERENCE_START',scene,len(points),'CPU',flush=True)
        with torch.inference_mode():output=model.predict(inputs,[sample])[0].pred_pts_seg
        prediction=result/'predictions.npz'
        np.savez_compressed(prediction,masks=output.pts_instance_mask[0],
            scores=output.instance_scores,labels=output.instance_labels,
            semantic=output.pts_semantic_mask[0])
        record=dict(scene=scene,profile='ESAM',frames=len(points),output=str(result),
            predicted_objects=len(output.instance_scores),seconds=time.monotonic()-start,
            prediction_sha256=sha(prediction),checkpoint_sha256=weights['checkpoint_sha256'],
            strict_weights_loaded=True,semantic_GT_in_inference=False,instance_GT_in_inference=False,
            mesh_geometry_segments_used=True,CPU_sparse_backend=True,CPU_pointops=adapter,CPU_feature_order=order_adapter,
            official_config_sha256=sha(config_path),architecture_and_inference_parameters_unchanged=True,
            import_compatibility=json.loads((WORK/'esam_import_compatibility.json').read_text()),
            coordinates='unaligned_scan_world',preparation=prepared)
        marker.write_text(json.dumps(record,indent=2)+'\n');rows.append(record)
        (WORK/'esam_results.json').write_text(json.dumps(dict(results=rows,complete=len(rows)==len(registration["scenes"])),indent=2)+'\n')
        print('ESAM_INFERENCE_COMPLETE',scene,len(output.instance_scores),round(record['seconds'],1),flush=True)
    (WORK/'esam_results.json').write_text(json.dumps(dict(results=rows,complete=True),indent=2)+'\n')
    print('ESAM_ALL_THREE_COMPLETE',flush=True)

if __name__=='__main__':main()
