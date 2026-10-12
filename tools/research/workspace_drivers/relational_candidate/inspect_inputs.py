import json
import numpy as np
from pathlib import Path
root=Path('/home/lewisliu/datasets/scannet-sg-processed/partaware_v19')
for dataset,scene in [('scannet','scene0019_00'),('hypersim','ai_001_010')]:
    source=root/dataset/scene
    print(dataset,scene,[p.name for p in source.iterdir() if p.is_dir()])
    for name in ['refined_instance','fine_frontend_cache']:
        paths=sorted((source/name).glob('*'))
        print(name,'count',len(paths),'first',[p.name for p in paths[:6]])
    for path in sorted((source/'refined_instance').glob('*_instance.json'))[:1]:
        rows=json.loads(path.read_text());print('FIELDS',path.name,[{k:v for k,v in r.items() if not isinstance(v,list) and k not in ['description']} for r in rows[:5]])
    g=json.loads((source/'topology_map.json').read_text())
    print('NODE_FIELDS',[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in list(g['object_nodes']['nodes'].values())[:5]])
    t=json.loads((source/'validated_object_tracks.json').read_text())
    first=next(iter(t.values()));print('TRACK_KEYS',list(first));print('OBS_KEYS',list(first.get('observations',[{}])[0]))
    for folder in ['sam3_cache','sam3_source_evidence','parts']:
        paths=sorted((source/folder).rglob('*'));print('CACHE',folder,[(str(p.relative_to(source)),p.stat().st_size) for p in paths[:6] if p.is_file()])
    paths=sorted((source/'refined_instance').glob('*_updated_instance.json'))
    print('UPDATED_FIELDS',paths[0].name,[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in json.loads(paths[0].read_text())[:5]])
    cache=next(iter(sorted((source/'sam3_cache').glob('*.npz'))))
    with np.load(cache,allow_pickle=False) as z:
        print('NPZ',cache.name,[(k,z[k].shape,str(z[k].dtype)) for k in z.files])
        rr=json.loads(str(z['records'].item()));print('RAW_RECORDS',[{k:v for k,v in r.items() if not isinstance(v,(list,dict))} for r in rr[:3]])
