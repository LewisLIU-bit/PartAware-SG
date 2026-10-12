"""Official ESAM SAM-H superpoints on the preregistered 125 RGB-D frames.

No semantic labels or GT instances are read. Mirrors the author's
data/scannet200-mv/load_scannet_mv_data.py: area-ordered SAM masks,
20,000 depth points and eight KMeans groups for uncovered RGB-XYZ points.
Only batching (16 SAM point prompts) is reduced to fit this 8GB GPU.
"""
import fcntl,hashlib,json,os,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from sklearn.cluster import KMeans
sys.path.insert(0,'/home/lewisliu/PartAware-SG/scannet/script/thirdparty/Grounded-Segment-Anything/segment_anything')
from segment_anything import build_sam,SamAutomaticMaskGenerator

WORK=Path(__file__).resolve().parent
BASE=Path('/home/lewisliu/baselines/ESAM')
OUT=BASE/'data/local_scan200'
registration=json.loads((WORK/'preregistration.json').read_text())

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

rows=[]
for scene in registration['scenes']:
    info=json.loads((WORK/(scene+'_input.json')).read_text())
    manifest_path=Path(info['manifest']);manifest=json.loads(manifest_path.read_text())
    dest=OUT/scene;dest.mkdir(parents=True,exist_ok=True)
    # Restartable chunks release the GPU for any ready construction job.
    for begin in range(0,len(manifest['frames']),10):
        chunk=manifest['frames'][begin:begin+10]
        pending=[r for r in chunk if not (dest/(r['frame_id']+'.npz')).exists()]
        if not pending:continue
        print('ESAM_SAM_WAIT_GPU',scene,begin,flush=True)
        with (WORK/'gpu.lock').open('a') as mutex, (Path.home()/'.cache/partaware-sg/gpu.lock').open('a') as shared:
            fcntl.flock(mutex,fcntl.LOCK_EX)
            fcntl.flock(shared,fcntl.LOCK_EX)
            generator=SamAutomaticMaskGenerator(build_sam(checkpoint=str(BASE/'data/sam_vit_h_4b8939.pth')).to('cuda'),points_per_batch=16)
            for row in pending:
                frame=row['frame_id'];start=time.monotonic()
                image=cv2.cvtColor(cv2.imread(str(manifest_path.parent/row['rgb'])),cv2.COLOR_BGR2RGB)
                image=cv2.resize(image,(640,480),interpolation=cv2.INTER_LINEAR)
                depth=cv2.imread(str(manifest_path.parent/row['depth']),cv2.IMREAD_UNCHANGED)
                assert depth.shape==(480,640)
                pose=np.loadtxt(manifest_path.parent/row['pose'])
                masks=sorted(generator.generate(image),key=lambda r:r['area'],reverse=True)
                groups=np.full((480,640),-1,dtype=np.int64)
                for group,mask in enumerate(masks):groups[mask['segmentation']]=group
                yy,xx=np.indices(depth.shape);valid=depth>0.1
                zz=depth[valid].astype(np.float64)/float(manifest['depth_scale'])
                # The official pretrained preprocessing uses this fixed sensor
                # intrinsic, independently of semantic/instance annotations.
                xyz=np.column_stack(((xx[valid]-319.5)*zz/577.870605,
                                     (yy[valid]-239.5)*zz/577.870605,zz))
                xyz=xyz@pose[:3,:3].T+pose[:3,3]
                points=np.column_stack((xyz,image[valid]))
                groups=groups[valid]
                rng=np.random.RandomState(0+int(row['source_frame_id']))
                selected=rng.choice(len(points),20000,replace=len(points)<20000)
                points=points[selected];groups=groups[selected]
                uncovered=groups==-1
                if uncovered.any():
                    count=min(8,int(uncovered.sum()))
                    groups[uncovered]=KMeans(n_clusters=count,n_init=10,random_state=0).fit_predict(points[uncovered])+groups.max()+1
                _,groups=np.unique(groups,return_inverse=True)
                partial=dest/(frame+'.partial.npz');final=dest/(frame+'.npz')
                np.savez_compressed(partial,points=points.astype(np.float32),superpoints=groups.astype(np.int64))
                partial.replace(final)
                print('ESAM_SAM_FRAME',scene,frame,'superpoints',len(np.unique(groups)),
                      'seconds',round(time.monotonic()-start,2),flush=True)
            del generator
            torch.cuda.empty_cache()
    frame_files=[dest/(r['frame_id']+'.npz') for r in manifest['frames']]
    assert all(p.exists() for p in frame_files)
    record=dict(scene=scene,frames=len(frame_files),manifest_sha256=sha(manifest_path),
        input_GT_used=False,frame_files=[dict(path=str(p),sha256=sha(p)) for p in frame_files],
        points_per_frame=20000,SAM='official SAM ViT-H',SAM_points_per_batch=16,
        uncovered_KMeans=8,random_seed_rule='source_frame_id',
        coordinates='unaligned_scan_world',intrinsic='official fixed 577.870605/319.5/239.5',
        author_preprocessing='data/scannet200-mv/load_scannet_mv_data.py')
    (dest/'preparation.json').write_text(json.dumps(record,indent=2)+'\n')
    rows.append(record)
    (WORK/'esam_prepared.json').write_text(json.dumps(dict(results=rows,complete=len(rows)==len(registration["scenes"])),indent=2)+'\n')
print('ESAM_INPUTS_PREPARED',flush=True)
