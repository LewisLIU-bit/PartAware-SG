from pathlib import Path
import sys,json,hashlib,fcntl,argparse
from contextlib import redirect_stdout
import io
import numpy as np
from PIL import Image
parser=argparse.ArgumentParser();parser.add_argument('--jobs',required=True);parser.add_argument('--cache-dir',required=True);args=parser.parse_args();jobfile=Path(args.jobs);config=json.loads(jobfile.read_text());repo=Path(config['repo']);sys.path.insert(0,str(repo/'scannet/script/pipeline_components'))
from sam3_worker import CHECKPOINT_SHA256,sha256
import torch
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
weights=repo/'checkpoints/sam3/sam3.pt';assert sha256(weights)==CHECKPOINT_SHA256
out=Path(args.cache_dir);out.mkdir(exist_ok=True);receipts=[]
with (Path.home()/'.cache/partaware-sg/gpu.lock').open('a') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX);loader=io.StringIO()
 with redirect_stdout(loader):model=build_sam3_image_model(checkpoint_path=str(weights),bpe_path=str(repo/'scannet/script/thirdparty/SAM3/sam3/assets/bpe_simple_vocab_16e6.txt.gz'),load_from_HF=False,device='cuda',enable_inst_interactivity=False)
 if 'missing_keys=' in loader.getvalue():raise ValueError('Incomplete pretrained parameters')
 processor=Sam3Processor(model,device='cuda',confidence_threshold=.5)
 for job in config['jobs']:
  image=Image.open(job['rgb']).convert('RGB');x0,y0,x1,y1=job['box'];box=[(x0+x1)/(2*image.width),(y0+y1)/(2*image.height),(x1-x0)/image.width,(y1-y0)/image.height]
  with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):state=processor.set_image(image);result=processor.add_geometric_prompt(box=box,label=True,state=state)
  masks=result['masks'][:,0].cpu().numpy().astype(bool);scores=result['scores'].float().cpu().numpy();signature=hashlib.sha256(json.dumps(dict(job=job,rgb_sha256=sha256(job['rgb']),checkpoint=CHECKPOINT_SHA256,worker=sha256(__file__)),sort_keys=True).encode()).hexdigest();file=f"{job['candidate']}_{job['frame_id']}.npz"
  np.savez_compressed(out/file,packed_masks=np.packbits(masks,axis=-1),shape=np.array(masks.shape),scores=scores,signature=signature)
  receipts.append(dict(**job,file=file,signature=signature,rgb_sha256=sha256(job['rgb']),scores=scores.tolist()));print('VISUAL',job['candidate'],job['frame_id'],scores.tolist(),flush=True)
  del state,result;torch.cuda.empty_cache()
(out/'audit.json').write_text(json.dumps(dict(ground_truth_used=False,gpt_api_calls=0,qwen_api_calls=0,checkpoint_sha256=CHECKPOINT_SHA256,frames=receipts),indent=2))
