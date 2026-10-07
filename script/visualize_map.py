import os
import sys
import argparse
from pathlib import Path

file_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(file_path))

from utils.result_visualization import visualize_map_with_nodes



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    example_scene = Path(__file__).resolve().parents[1] / "sample_data/scans/scene0000_00"
    parser.add_argument("--map_ply_path", type=str, default=str(example_scene / "instance_cloud_cleaned.ply"))
    parser.add_argument("--topology_map_path", type=str, default=str(example_scene / "topology_map.json"))
    parser.add_argument("--node_radius", type=float, default=0.1)
    parser.add_argument("--show_bboxes", action="store_true")
    parser.add_argument("--show_edges", action="store_true")
    parser.add_argument("--enable_picking", action="store_true",
                        help="Enable node-center picking and print picked object names. Use 'Shift' + 'Left Click' on a node sphere to print object name")
    parser.add_argument("--show_parts", action="store_true", help="Add part child nodes to the original map scene")
    parser.add_argument("--show_part_points", action="store_true", help="Also display part point clouds")
    parser.add_argument("--include_provisional_parts", action="store_true")
    parser.add_argument("--part_radius", type=float, default=0.035)
    parser.add_argument("--check_only", action="store_true")
    parser.add_argument("--screenshot", default=None, help="Save a preview without an interactive window")
    parser.add_argument('--view-front', nargs=3, type=float, default=None)
    parser.add_argument('--view-lookat', nargs=3, type=float, default=None)
    parser.add_argument('--view-zoom', type=float, default=None)
    args = parser.parse_args()

    example_map_ply_path = args.map_ply_path
    example_topology_map_path = args.topology_map_path

    # Check if example files exist before running
    if os.path.exists(example_map_ply_path) and os.path.exists(example_topology_map_path):
        try:
            tracking_colors = visualize_map_with_nodes(
                map_ply_path=example_map_ply_path,
                topology_map_path=example_topology_map_path,
                node_radius=args.node_radius,
                show_bboxes=args.show_bboxes,
                show_edges=args.show_edges,
                enable_picking=args.enable_picking,
                show_parts=args.show_parts,
                show_part_points=args.show_part_points,
                include_provisional_parts=args.include_provisional_parts,
                part_radius=args.part_radius,
                check_only=args.check_only,
                screenshot_path=args.screenshot,
                view_front=args.view_front,
                view_lookat=args.view_lookat,
                view_zoom=args.view_zoom,
            )
            print(f"Successfully visualized map with {len(tracking_colors) if tracking_colors else 0} tracking IDs")
        except Exception as e:
            print(f"Error during map visualization: {e}",file=sys.stderr)
            sys.exit(1)
    else:
        print("Example files not found. Please modify the paths above to match your actual files.")
        print(f"Map PLY path: {example_map_ply_path}")
        print(f"Topology map path: {example_topology_map_path}")
    
    print("\nVisualization example complete!")
    print("Modify the file paths above to use with your actual data.")
