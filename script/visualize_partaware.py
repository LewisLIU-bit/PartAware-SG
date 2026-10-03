"""Compatibility entry point for part overlays in the original map renderer."""
import argparse
from pathlib import Path

from utils.result_visualization import visualize_map_with_nodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--base-cloud', help='Original tracking-ID PLY; otherwise infer from the part run configuration')
    parser.add_argument('--include-provisional', action='store_true')
    parser.add_argument('--nodes-only', action='store_true', help='Compatibility flag; part clouds are hidden by default')
    parser.add_argument('--show-part-points', action='store_true')
    parser.add_argument('--show-object-edges', action='store_true')
    parser.add_argument('--object-radius', type=float, default=0.07)
    parser.add_argument('--part-radius', type=float, default=0.035)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--enable-picking', action='store_true')
    parser.add_argument('--screenshot', default=None)
    args = parser.parse_args()
    if not args.base_cloud:
        import json
        graph_path = Path(args.graph)
        candidates = [graph_path.parent/'instance_cloud.ply', graph_path.parent.parent/'instance_cloud.ply']
        config = graph_path.parent/'run_config.json'
        if config.is_file():
            source = json.loads(config.read_text()).get('processed_scene')
            if source:
                candidates.append(Path(source)/'instance_cloud.ply')
        args.base_cloud = next((str(p) for p in candidates if p.is_file()), None)
        if args.base_cloud is None:
            parser.error('Cannot locate the original map PLY; specify --base-cloud')
    for value in [args.graph, args.base_cloud]:
        if not Path(value).is_file():
            parser.error(f'Input does not exist: {value}')
    if args.object_radius <= 0 or args.part_radius <= 0:
        parser.error('Node radii must be positive')
    visualize_map_with_nodes(
        args.base_cloud, topology_map_path=args.graph,
        node_radius=args.object_radius, part_radius=args.part_radius,
        show_bboxes=True, show_edges=args.show_object_edges,
        show_parts=True, show_part_points=args.show_part_points and not args.nodes_only,
        include_provisional_parts=args.include_provisional,
        enable_picking=args.enable_picking, check_only=args.check_only,
        screenshot_path=args.screenshot)


if __name__ == '__main__':
    main()
