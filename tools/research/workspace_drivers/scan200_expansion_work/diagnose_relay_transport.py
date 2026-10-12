"""Read-only relay reachability; no image generation, keys never logged."""
import json,sys
from pathlib import Path
import httpx
from openai import OpenAI
sys.path.insert(0,'/home/lewisliu/PartAware-SG/scannet/script')
from vision_api import Settings
WORK=Path(__file__).resolve().parent;s=Settings.load('/home/lewisliu/PartAware-SG/.gpt-vision.local.json')
rows=[]
for label,proxy in [('direct',None),('existing_local_proxy','http://127.0.0.1:17890')]:
 try:
  with httpx.Client(proxy=proxy,trust_env=False,timeout=25) as transport:
   client=OpenAI(api_key=s.api_key,base_url=s.base_url,http_client=transport,max_retries=0)
   models=client.models.list();ids=[m.id for m in models]
   row=dict(transport=label,success=True,configured_model_listed=s.model in ids,models_count=len(ids))
 except Exception as error:
  causes=[];cause=error
  while cause is not None and len(causes)<4:
   causes.append(dict(type=type(cause).__name__,message=s.redact(str(cause))[:500]));cause=cause.__cause__
  row=dict(transport=label,success=False,error_chain=causes)
 rows.append(row);print('RELAY_TRANSPORT_CHECK',json.dumps(row),flush=True)
(WORK/'relay_transport_diagnosis.json').write_text(json.dumps(dict(complete=True,image_calls=0,secret_values_recorded=False,results=rows),indent=2)+'\n')
