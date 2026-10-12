"""GT-free four-domain relational proposal screen from frozen raw caches.

One domain and one original frame at a time; no model, API, GPU, GT, new
coordinates or guessed gravity. The independent evaluation is a separate file.
"""
import gc,hashlib,json,resource,sys,time
from pathlib import Path
import cv2,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path[:0]=['/home/lewisliu/PartAware-SG/scannet/script',str(ROOT/'v20_work/research')]
from partaware.geometry import load_capture
from projection_verification import project
from relational_admission import POLICY,frame_relations,cannot_link,admit_peers
from stream_core import intersections,propose,smooth

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def save_masks(path,ownership,scores):
    masks=np.empty((len(scores),len(ownership)),bool)
    for rank in range(len(scores)):masks[rank]=ownership==rank
    np.savez_compressed(path,masks=masks,scores=scores)
    del masks;gc.collect()

def main():
    started=time.time();registration=json.loads((HERE/'hypothesis_registration.json').read_text())
    tests=json.loads((HERE/'behavior_test_receipt.json').read_text());assert tests['passed']
    results=[]
    for item in registration['items']:
        scene=item['scene'];source=Path(item['source']);output=HERE/'outputs'/item['dataset']/scene
        output.mkdir(parents=True,exist_ok=True)
        for field,digest in item['input_hashes'].items():assert sha(item[field])==digest
        if item['domain_kind']=='unlabeled_mesh':xyz=np.asarray(o3d.io.read_triangle_mesh(item['domain']).vertices)
        else:
            with np.load(item['domain'],allow_pickle=False) as z:xyz=z['xyz']
        with np.load(item['peer'],allow_pickle=False) as z:peer=z['masks'];peer_scores=z['scores']
        assert peer.shape==(len(peer_scores),len(xyz)) and peer.dtype==bool
        graph=json.loads((source/'topology_map.json').read_text());keys=list(graph['object_nodes']['nodes']);index={key:i for i,key in enumerate(keys)}
        cloud_name=graph['geometry_provenance']['source'];assert Path(cloud_name).name==cloud_name
        cloud_path=source/cloud_name;cloud=o3d.io.read_point_cloud(str(cloud_path));points=np.asarray(cloud.points)
        rgb=np.rint(np.asarray(cloud.colors)*255).astype(np.int64);ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
        distance,nearest=cKDTree(points).query(xyz,workers=2);labels=ids[nearest];labels[distance>.05]=-1
        segments=None
        if item['segments']:
            segments=np.asarray(json.loads(Path(item['segments']).read_text())['segIndices']);labels=smooth(labels,segments)
        owner=np.full(len(xyz),-1,np.int16)
        for i,key in enumerate(keys):owner[labels==int(key)]=i
        del cloud,points,rgb,ids,distance,nearest,labels;gc.collect()
        tracks_path=source/'validated_object_tracks.json';tracks=json.loads(tracks_path.read_text())
        native_scores=np.asarray([tracks.get(k,{}).get('confidence',graph['object_nodes']['nodes'][k].get('confidence',0)) for k in keys])
        # Only explicit object/part parent links can veto a split interpretation.
        # No class words are used to guess that two shapes must be parts.
        parts=graph.get('part_nodes',{});part_pairs=[]
        if isinstance(parts,dict):
            for pid,part in parts.items():
                parent=str(part.get('parent_id',''))
                if str(pid) in index and parent in index and str(pid)!=parent:part_pairs.append((index[str(pid)],index[parent]))
        _,jobs,kd,kc,scale=load_capture(item['manifest']);n=len(keys)
        separate_frames=np.zeros((len(jobs),n,n),bool);whole_frames=np.zeros_like(separate_frames)
        camera_centers=[];frame_inputs=[];frame_statistics=[]
        for f,job in enumerate(jobs):
            fid=job['frame_id'];png=source/'refined_instance'/f'{fid}.png';cache=source/'sam3_cache'/f'{fid}.npz'
            pose=np.loadtxt(job['pose']);depth=cv2.imread(str(job['depth']),cv2.IMREAD_UNCHANGED)
            original=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
            if original is None or depth is None or not cache.is_file():raise ValueError('Frozen original mask/depth/raw SAM cache missing')
            ix,uv=project(xyz,pose,kd,kc,depth,scale,*original.shape[:2])
            with np.load(cache,allow_pickle=False) as z:
                packed=z['packed_masks'];shape=z['shape'];records=json.loads(str(z['records'].item()))
                assert tuple(shape[1:])==original.shape and len(records)==int(shape[0])
                # Read original bits directly at projected pixels; do not
                # expand every cached image mask or store a point/view matrix.
                shifts=(7-uv[:,0]%8).astype(np.uint8)
                sampled=((packed[:,uv[:,1],uv[:,0]//8]>>shifts)&1).astype(bool)
            frame_owner=owner[ix];frame_labels=original[uv[:,1],uv[:,0]].astype(np.int64)
            separate,whole,joint=frame_relations(frame_owner,frame_labels,sampled,n,part_pairs)
            separate_frames[f]=separate;whole_frames[f]=whole;camera_centers.append(pose[:3,3].tolist())
            frame_statistics.append(dict(frame_id=fid,visible_domain_points=len(ix),
                original_raw_masks=len(records),separated_identity_pairs=int(np.triu(separate,1).sum()),
                ambiguous_common_whole_pairs=int(np.triu(whole,1).sum())))
            frame_inputs.append(dict(frame_id=fid,depth_sha256=sha(job['depth']),pose_sha256=sha(job['pose']),
                original_mask_sha256=sha(png),raw_SAM_cache_sha256=sha(cache)))
            del sampled,packed,ix,uv,depth,original,frame_owner,frame_labels
            if (f+1)%10==0 or f+1==len(jobs):print('RELATION_FRAME',scene,f+1,'/',len(jobs),flush=True)
        relations,proof=cannot_link(separate_frames,whole_frames,camera_centers)
        cross,mass=intersections(owner,n,peer);admitted,deferrals=admit_peers(cross,mass,relations)
        print('RELATIONAL_ADMISSION',scene,int(admitted.sum()),'of',len(peer),'relations',len(proof),flush=True)
        results_for_scene=[]
        for mode,indices in [('signed_CORE_V_control',np.arange(len(peer))),('relational_CORE_V',np.flatnonzero(admitted))]:
            # Indexing a boolean matrix copies it. The full control keeps a
            # view of the signed peer; the reduced candidate makes one copy.
            chosen=peer if mode=='signed_CORE_V_control' else peer[indices]
            ownership,scores,audit=propose(owner,native_scores,chosen,peer_scores[indices],xyz)
            if segments is not None:ownership=smooth(ownership,segments)
            path=output/(mode+'_masks.npz');save_masks(path,ownership,scores)
            results_for_scene.append(dict(mode=mode,masks_file=str(path),masks_sha256=sha(path),
                peer_indices=indices.tolist(),native_keys=keys,audit=audit))
            del chosen,ownership,scores;gc.collect()
        row=dict(scene=scene,dataset=item['dataset'],domain_points=len(xyz),source=str(source),
            native_proposals=n,peer_proposals=len(peer),admitted_peer=int(admitted.sum()),
            cannot_link_pairs=len(proof),relation_proof=proof,deferrals=deferrals,
            frame_statistics=frame_statistics,frame_inputs=frame_inputs,
            signed_input_hashes=item['input_hashes'],source_cloud_sha256=sha(cloud_path),
            graph_sha256=sha(source/'topology_map.json'),tracks_sha256=sha(tracks_path),
            explicit_part_pair_indices=part_pairs,results=results_for_scene)
        results.append(row)
        result=dict(complete=False,GT_in_construction=False,full_pipeline_replay=False,
            API_calls=0,training=False,GPU=False,generated_points=0,physical_points_deleted=0,
            policy=POLICY,results=results,source_hashes={p.name:sha(p) for p in [HERE/'build_screen.py',HERE/'stream_core.py',HERE/'relational_admission.py']})
        (HERE/'construction.json').write_text(json.dumps(result,indent=2)+'\n')
        del xyz,peer,owner,cross,mass,separate_frames,whole_frames;gc.collect()
    result['complete']=True;result['elapsed_seconds']=time.time()-started
    result['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (HERE/'construction.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RELATIONAL_FOUR_DOMAIN_SCREEN_COMPLETE',result['elapsed_seconds'],result['peak_RSS_MiB'],flush=True)
if __name__=='__main__':main()
