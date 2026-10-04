"""Make VLPart nodes and hierarchy part of the canonical construction output."""
import json
from pathlib import Path


def construct(context):
    script = context.repo / 'scannet/script/run_partaware.py'
    python = context.repo / '.venv-vlpart/bin/python'
    parts = context.scene / 'parts'
    command = [str(python), str(script), '--processed-scene', str(context.scene),
               '--output', str(parts), '--image-size', '480', '--dbscan-eps', '.05',
               '--subtract-contained', '--sam-checkpoint',
               str(context.repo / 'scannet/script/thirdparty/Grounded-Segment-Anything/sam_vit_h_4b8939.pth')]
    command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
    context.execute(command, 'VLPart 部件识别与 OP3DSG 部件融合')
    graph = json.loads((parts / 'partaware_graph.json').read_text())
    nodes = graph['object_nodes']['nodes']
    known_parts = {key: value for key, value in graph['part_nodes'].items() if value['status'] == 'confirmed'}
    # Keep heterogeneous embeddings separate while exposing a single queryable graph.
    edges = [edge for hypothesis in (graph.get('edge_hypotheses') or {}).values()
             for edge in (hypothesis.get('edges') or {}).values()]
    graph['scene_graph'] = {'schema_version': 1,
        'nodes': {**{key: {'id': key, 'name': value['name'], 'node_type': 'object'} for key, value in nodes.items()},
                  **known_parts}, 'edges': edges + graph['part_relations']}
    graph['partaware_provenance']['canonical_output'] = 'topology_map.json'
    (context.scene / 'topology_map.json').write_text(json.dumps(graph, indent=2) + '\n')
    context.event('部件已写入正式场景图', object_nodes=len(nodes), confirmed_part_nodes=len(known_parts),
                  part_of_edges=len(graph['part_relations']), provisional_parts=len(graph['part_nodes']) - len(known_parts))
