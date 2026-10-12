"""Register identical image-confirmation policy on four frozen input domains."""
import hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'v20_work'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text());items=[]
for scene in reg['registration']['scenes']:
 source=next(r['processed_scene'] for r in reg['results'] if r['scene_id']==scene and r['profile']=='v19')
 manifest=json.loads((ROOT/'scan200_work'/f'{scene}_input.json').read_text())['manifest']
 raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
 peer=Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/projection_verification/scannet')/scene/'verified_text_positive_peer_masks.npz'
 domain=raw/f'{scene}_vh_clean_2.ply';segment=raw/f'{scene}_vh_clean_2.0.010000.segs.json'
 items.append(dict(dataset='scannet',scene=scene,source=source,manifest=manifest,domain=str(domain),domain_kind='unlabeled_mesh',segments=str(segment),peer=str(peer)))
registration=json.loads((ROOT/'esam_hypersim_work/registration.json').read_text())
items.append(dict(dataset='hypersim',scene=registration['scene'],source='/home/lewisliu/datasets/scannet-sg-processed/partaware_v19/hypersim/'+registration['scene'],manifest=registration['manifest'],domain=str(ROOT/'esam_hypersim_work/observed_domain.npz'),domain_kind='observed_rgbd_voxels',segments=None,peer=str(ROOT/'esam_hypersim_work/ESAM_common_domain_masks.npz')))
for r in items:
 r['input_hashes']={k:sha(r[k]) for k in ['manifest','domain','peer']}
 if r['segments']:r['input_hashes']['segments']=sha(r['segments'])
payload=dict(complete=True,GT_in_construction=False,policy='peer formal proposal requires >=2 independent original image masks, depth visibility and consistent name; unconfirmed remains provisional geometry',items=items)
(WORK/'four_domain_confirmation_registration.json').write_text(json.dumps(payload,indent=2)+'\n')
print('FOUR_DOMAINS_SAME_POLICY_REGISTERED',flush=True)
