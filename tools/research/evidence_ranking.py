"""Untrained quality ranking from independent measured surface support.

This changes ranking only, never existence, masks, point ownership or geometry.
No GT, category or scene-specific rule. It is not a calibrated IoU predictor.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def quality_score(confidence,confirmed_fraction):
    confidence=float(confidence)
    if not np.isfinite(confidence) or not 0<=confidence<=1:
        raise ValueError('Confidence must be finite in [0,1]')
    if confirmed_fraction is None:return confidence
    fraction=float(confirmed_fraction)
    if not np.isfinite(fraction) or not 0<=fraction<=1:
        raise ValueError('Measured surface fraction must be in [0,1]')
    return confidence*np.sqrt(fraction)

def construct(context):
    scene=Path(context.scene);tracks_path=scene/'validated_object_tracks.json'
    if not tracks_path.exists():tracks_path=scene/'object_tracks.json'
    tracks=json.loads(tracks_path.read_text());records=[]
    graph_path=scene/'topology_map.json';graph=json.loads(graph_path.read_text())
    source=scene/graph['geometry_provenance']['source']
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    for nid,track in tracks.items():
        old=track['confidence'];fraction=track.get('multi_view_point_fraction')
        value=float(quality_score(old,fraction));track.update(confidence=value,
          detector_observation_confidence=old,quality_ranking_type='uncalibrated_measured_support_geometric_mean')
        records.append(dict(node=nid,old_confidence=old,confirmed_fraction=fraction,new_confidence=value))
        if nid in graph['object_nodes']['nodes']:
            graph['object_nodes']['nodes'][nid]['confidence']=value
    tracks_path.write_text(json.dumps(tracks,indent=2)+'\n')
    graph['quality_ranking_provenance']=dict(module='evidence_ranking',trained=False,
        GT_used=False,nodes_removed=0,coordinates_changed=False,geometry_sha256=before,
        unknown_fraction_preserves_original_score=True)
    graph_path.write_text(json.dumps(graph,indent=2)+'\n')
    parts=scene/'parts/partaware_graph.json'
    if parts.exists():parts.write_text(json.dumps(graph,indent=2)+'\n')
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    (scene/'quality_ranking_audit.json').write_text(json.dumps(dict(records=records,
        geometry_sha256=before,GT_used=False,trained=False),indent=2)+'\n')
    context.event('Independent measured-surface quality ranking',nodes=len(records),
        nodes_removed=0,geometry_changed=False)
