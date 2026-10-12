"""Frozen SAM3 box-only cross-view evidence; no GT, labels or image API."""
import argparse,contextlib,fcntl,hashlib,io,json,os,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];REPO=Path('/home/lewisliu/PartAware-SG')
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.path.insert(0,str(REPO/'scannet/script/thirdparty/SAM3'))
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
weights=REPO/'checkpoints/sam3/sam3.pt';expected='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'
assert sha(weights)==expected
lock=Path.home()/'.cache/partaware-sg/gpu.lock'
summary=json.loads((ROOT/'v20_work/projection_preparation.json').read_text());receipts=[]
p=argparse.ArgumentParser();p.add_argument('--text',action='store_true');args=p.parse_args()
with lock.open('a') as stream:
 fcntl.flock(stream,fcntl.LOCK_EX)
 log=io.StringIO()
 with contextlib.redirect_stdout(log):
  model=build_sam3_image_model(checkpoint_path=str(weights),bpe_path=str(REPO/'scannet/script/thirdparty/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),load_from_HF=False,device='cuda',enable_inst_interactivity=False)
 if 'missing_keys=' in log.getvalue():raise RuntimeError('Incomplete frozen checkpoint loading')
 processor=Sam3Processor(model,device='cuda',confidence_threshold=.5)
 for item in summary['results']:
  target=Path(item['target']);path=target/('projection_text_jobs.json' if args.text else 'projection_jobs.json');jobs=json.loads(path.read_text());cache=target/('sam3_text_box_cache' if args.text else 'sam3_box_cache');cache.mkdir(exist_ok=True)
  rows=[]
  for job in jobs['jobs']:
   if sha(job['rgb'])!=job['rgb_sha256']:raise ValueError('RGB cache input changed')
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    state=processor.set_image(Image.open(job['rgb']).convert('RGB'))
   for prompt in job['prompts']:
    processor.reset_all_prompts(state)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
     if args.text and prompt.get('text'):processor.set_text_prompt(prompt['text'],state)
     output=processor.add_geometric_prompt(prompt['box'],True,state)
    masks=output['masks'].squeeze(1).detach().cpu().numpy();scores=output['scores'].detach().float().cpu().numpy()
    name=f"{job['frame_id']}_{prompt['peer_index']}.npz";cp=cache/name
    np.savez_compressed(cp,masks=masks,scores=scores)
    rows.append(dict(frame_id=job['frame_id'],peer_index=prompt['peer_index'],cache=name,mask_candidates=len(masks),sha256=sha(cp)))
   del state;torch.cuda.empty_cache()
   print('SAM3_BOX_VERIFICATION_FRAME',jobs['scene'],job['frame_id'],len(job['prompts']),flush=True)
  receipt=dict(complete=True,scene=jobs['scene'],jobs_sha256=sha(path),weights_sha256=expected,code_sha256=sha(__file__),
               GT_in_construction=False,text_prompt='cached image-derived name when geometrically supported, else visual' if args.text else 'native visual-only dummy, no object names',new_GPT_calls=0,training=False,rows=rows)
  (target/('sam3_text_box_receipt.json' if args.text else 'sam3_box_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n');receipts.append(receipt)
(ROOT/'v20_work'/('projection_text_inference.json' if args.text else 'projection_inference.json')).write_text(json.dumps(dict(complete=True,full_geometry_replay=False,results=receipts),indent=2)+'\n')
