"""Ensure recovered geometry still loads through the original graph API."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import open3d as o3d
from pipeline_components.canonical_geometry import publish

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'script/include'))
from topology_map import TopologyMap


class PublicationContractTests(unittest.TestCase):
    def test_new_measured_object_loads_in_original_viewer(self):
        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            p = np.array([[x,y,z] for x in [0,.1,.2] for y in [0,.1,.2] for z in [0,.1,.2]])
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
            cloud.colors = o3d.utility.Vector3dVector(np.tile([7/255,0,0],(len(p),1)))
            o3d.io.write_point_cloud(str(scene/'instance_cloud_cleaned.ply'),cloud)
            node = {'name':'cabinet','visual_embedding':[0.]*256,'text_embedding':[0.]*384}
            graph = {'object_nodes':{'nodes':{'7':node}},'scene_graph':{'nodes':{},'edges':[]}}
            (scene/'topology_map_cleaned.json').write_text(json.dumps(graph))
            context = SimpleNamespace(scene=scene,graph_geometry=scene/'instance_cloud_cleaned.ply',
                manifest=None,edge_threshold=2,event=lambda *a,**k:None)
            publish(context)
            current = json.loads((scene/'topology_map.json').read_text())
            loaded = TopologyMap()
            loaded.read_from_json(json.dumps(current))
            self.assertEqual(loaded.get_entity('7').id,'7')
            self.assertEqual(current['scene_graph']['nodes']['7']['shape'],current['object_nodes']['nodes']['7']['shape'])


if __name__=='__main__':unittest.main()
