"""Read-only retrieval of pinned author source; it is not run or modified."""
import hashlib,json,urllib.request
from pathlib import Path
root=Path(__file__).resolve().parent
def get(url):
    r=urllib.request.Request(url,headers={'User-Agent':'EviScene-research-audit'})
    return urllib.request.urlopen(r,timeout=30).read()
api='https://api.github.com/repos/PKU-EPIC/MaskClustering'
repo=json.loads(get(api));head=json.loads(get(api+'/commits/'+repo['default_branch']));commit=head['sha']
tree=json.loads(get(api+'/git/trees/'+commit+'?recursive=1'))
files=[r['path'] for r in tree['tree'] if r['path'].endswith('.py')]
print('FILES',json.dumps(files))
rows=[]
for path in files:
    if any(s in path.lower() for s in ['graph','cluster','mask','consensus']):
        raw=get('https://raw.githubusercontent.com/PKU-EPIC/MaskClustering/'+commit+'/'+path)
        destination=root/'official_source'/path;destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(raw)
        rows.append(dict(path=path,sha256=hashlib.sha256(raw).hexdigest()))
        print('SOURCE',path,raw.decode()[:16000])
result=dict(official_repository=repo['html_url'],commit=commit,files=rows,
            source_inspection_only=True,author_pipeline_not_reproduced=True)
(root/'official_source_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
