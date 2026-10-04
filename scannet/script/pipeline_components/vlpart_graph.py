"""Make VLPart nodes and hierarchy part of the canonical construction output."""
import json
from pathlib import Path
import shutil
import tempfile


def construct(context):
    script = context.repo / 'scannet/script/run_partaware.py'
    python = context.repo / '.venv-vlpart/bin/python'
    parts = context.scene / 'parts'
    if parts.is_symlink():
        raise ValueError('Part output must not be a symlink')
    previous_log = (parts/'run_zh.jsonl').read_text() if (parts/'run_zh.jsonl').exists() else ''
    # Rebuild in a fresh sibling directory and publish only a completed result.
    with tempfile.TemporaryDirectory(prefix='.parts-rebuild-', dir=context.scene) as temporary:
        fresh = Path(temporary)
        command = [str(python), str(script), '--processed-scene', str(context.scene),
                   '--output', str(fresh), '--image-size', '480', '--dbscan-eps', '.05',
                   '--subtract-contained', '--sam-checkpoint',
                   str(context.repo / 'scannet/script/thirdparty/Grounded-Segment-Anything/sam_vit_h_4b8939.pth')]
        command += ['--manifest', str(context.manifest)] if context.manifest else ['--image-dir', str(context.image_dir)]
        context.execute(command, 'VLPart 部件识别与 OP3DSG 部件融合')
        graph = json.loads((fresh/'partaware_graph.json').read_text())
        graph['partaware_provenance']['output'] = str(parts)
        configuration = json.loads((fresh/'run_config.json').read_text())
        configuration['output'] = str(parts)
        (fresh/'run_config.json').write_text(json.dumps(configuration, indent=2)+'\n')
        (fresh/'partaware_graph.json').write_text(json.dumps(graph, indent=2)+'\n')
        if previous_log:
            boundary = json.dumps({'message': '重新构建部件并同步当前物体父节点'}, ensure_ascii=False)+'\n'
            (fresh/'run_zh.jsonl').write_text(previous_log+boundary+(fresh/'run_zh.jsonl').read_text())
        if parts.exists():
            if parts.resolve().parent != context.scene.resolve():
                raise ValueError('Part cleanup escaped the processed scene')
            shutil.rmtree(parts)
        shutil.move(str(fresh), str(parts))
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
