"""Validate physical direction and repeated-instance hypotheses without GT."""
import unittest
import json
import tempfile
from pathlib import Path
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

from pipeline_components.plane_boxes import fit
from pipeline_components.canonical_geometry import fit_box
from pipeline_components.repeated_instances import fit_repetition, eligible, free_space
from pipeline_components.visual_part_anchoring import acceptance as part_acceptance
from types import SimpleNamespace
from pipeline_components.fine_boundary import supported_samples
from pipeline_components.part_body_assembly import boundary_relation


class V13GeometryChecks(unittest.TestCase):
    def test_visual_part_publication_preserves_sources_and_unrelated_geometry(self):
        from pipeline_components.visual_part_anchoring import construct
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]/'script/include'))
        from topology_map import TopologyMap
        with tempfile.TemporaryDirectory() as directory:
            scene=Path(directory);(scene/'parts').mkdir();(scene/'refined_instance').mkdir()
            y,z=np.meshgrid(np.linspace(-.5,.5,12),np.linspace(-.5,.5,12))
            panel=np.column_stack([np.full(y.size,.2),y.ravel(),z.ravel()])
            body=np.concatenate([panel,panel*np.array([-1,1,1])])
            other=body+np.array([3.,0.,0.])
            generated=panel+np.array([-.01,0.,0.])
            def save(name, regions):
                p=np.concatenate(list(regions.values()));colors=np.concatenate([
                    np.tile([int(gid)/255,0.,0.],(len(q),1)) for gid,q in regions.items()])
                cloud=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
                cloud.colors=o3d.utility.Vector3dVector(colors)
                o3d.io.write_point_cloud(str(scene/name),cloud)
            save('instance_cloud_cleaned.ply',{'1':panel,'2':body,'3':other})
            save('instance_cloud_completed.ply',{'1':np.concatenate([panel,generated]),'2':body,'3':other})
            nodes={gid:dict(id=gid,name=name,visual_embedding=[0.]*256,text_embedding=[0.]*384)
                   for gid,name in [('1','door'),('2','cabinet'),('3','cabinet')]}
            part=dict(id='p',name='cabinet: door',status='confirmed',parent_id='2',
                parent_evidence={'2':8},parent_evidence_frames=10,observed_frames=list(range(10)))
            graph=dict(object_nodes={'nodes':nodes},part_nodes={'p':part,'q':{**part,'id':'q','status':'provisional'}},
                part_relations=[],scene_graph={'nodes':{gid:{**n,'node_type':'object'} for gid,n in nodes.items()},'edges':[]})
            (scene/'topology_map.json').write_text(json.dumps(graph));np.save(scene/'parts/p.points.npy',panel)
            tracks={gid:dict(confidence=.9,observed_frames=list(range(10)),observations=[]) for gid in nodes}
            (scene/'validated_object_tracks.json').write_text(json.dumps(tracks))
            local=scene/'refined_instance/test_updated_instance.json'
            local.write_text(json.dumps([{'instance_id':1,'frame_instance_id':1}]))
            context=SimpleNamespace(scene=scene,graph_geometry=scene/'instance_cloud_completed.ply',
                manifest=None,edge_threshold=2,event=lambda *a,**k:None)
            construct(context)
            result=json.loads((scene/'topology_map.json').read_text())
            self.assertEqual(set(result['object_nodes']['nodes']),{'2','3'})
            self.assertNotIn('1',result['scene_graph']['nodes'])
            self.assertNotIn('q',result['scene_graph']['nodes'])
            self.assertEqual(result['part_nodes']['owned_1_part']['parent_id'],'2')
            np.testing.assert_array_equal(np.load(scene/'parts/owned_1_part.points.npy'),panel)
            for name in ['instance_cloud_cleaned.ply','instance_cloud_completed.ply']:
                cloud=o3d.io.read_point_cloud(str(scene/name));p=np.asarray(cloud.points)
                ids=np.rint(np.asarray(cloud.colors)[:,0]*255).astype(int)
                np.testing.assert_array_equal(p[ids==3],other)
                parent=p[ids==2]
                self.assertTrue(np.all(cKDTree(parent).query(panel)[0]==0))
                if 'completed' in name:self.assertTrue(np.all(cKDTree(parent).query(generated)[0]==0))
            self.assertEqual(json.loads(local.read_text())[0]['instance_id'],2)
            loaded=TopologyMap();loaded.read_from_json(json.dumps(result))
            self.assertEqual(loaded.get_entity('2').id,'2')

    def test_two_partial_orthogonal_faces_recover_rotated_physical_axes(self):
        u, z = np.meshgrid(np.linspace(0, 1, 31), np.linspace(0, 2, 41))
        a = np.column_stack([np.zeros(u.size), u.ravel(), z.ravel()])
        b = np.column_stack([u.ravel()*.7, np.zeros(u.size), z.ravel()])
        rotation = Rotation.from_euler('z', .37).as_matrix()
        points = np.concatenate([a, b]) @ rotation.T
        center, shape, evidence = fit(points, True, fit_box)
        self.assertTrue(evidence['accepted'])
        self.assertAlmostEqual(evidence['yaw_radians'], .37, places=3)
        self.assertAlmostEqual(sorted([shape['length'],shape['width']])[0], .7, places=3)
        self.assertTrue(evidence['point_coordinates_preserved'])

    def test_one_plane_cannot_supply_unknown_body_direction_or_thickness(self):
        x, z = np.meshgrid(np.linspace(0, 1, 25), np.linspace(0, 1, 25))
        points = np.column_stack([x.ravel(), np.zeros(x.size), z.ravel()])
        original = fit_box(points, True)
        center, shape, evidence = fit(points, True, fit_box)
        self.assertFalse(evidence['accepted'])
        self.assertEqual(shape, original[1])

    def rims(self, heights, views=8):
        rng = np.random.default_rng(8)
        p, v = [], []
        for height in heights:
            for view in range(views):
                q = rng.uniform(-.15, .15, (80, 3));q[:,2] = height+rng.normal(0,.00015,80)
                side = np.arange(80)%4
                q[side == 0,0] = -.15; q[side == 1,0] = .15
                q[side == 2,1] = -.15; q[side == 3,1] = .15
                p.append(q);v.append(np.full(80,view))
        return np.concatenate(p), np.concatenate(v)

    def test_repeated_rims_define_count_from_measurements(self):
        points, views = self.rims(np.arange(6)*.007+1.)
        layers, audit = fit_repetition(points, views)
        self.assertEqual(len(layers), 6)
        self.assertTrue(audit['accepted'])
        self.assertAlmostEqual(audit['period_m'], .007, places=3)

    def test_shelf_faces_and_irregular_texture_do_not_create_atomic_objects(self):
        for heights in [[1.,1.02], [1.,1.005,1.027,1.043]]:
            points, views = self.rims(heights)
            self.assertFalse(fit_repetition(points, views)[1]['accepted'])

    def test_single_view_repeated_texture_is_not_multiview_evidence(self):
        points, views = self.rims(np.arange(6)*.007, views=1)
        self.assertFalse(fit_repetition(points, views)[1]['accepted'])

    def test_inner_basin_contours_do_not_supply_outer_instance_boundaries(self):
        points, views = self.rims(np.arange(6)*.007)
        points[:,:2] *= .2
        footprint = np.array([[-.15,-.15,0.],[-.15,.15,0.],[.15,-.15,0.],[.15,.15,0.]])
        self.assertFalse(fit_repetition(points, views, footprint=footprint)[1]['accepted'])

    def test_dense_sampling_requires_distinct_views_and_retains_original_samples(self):
        points = np.array([[0.,0.,0.], [.0001,0.,0.], [.0002,0.,0.], [1.,1.,1.]])
        kept, counts = supported_samples(points, np.array([0,1,2,0]))
        np.testing.assert_array_equal(kept, points[:1])
        self.assertEqual(counts.tolist(), [3])

    def test_raster_edge_is_not_free_space_but_floating_surface_is_rejected(self):
        depth=np.full((16,16),1000,np.uint16);depth[8,8]=3000
        views=SimpleNamespace(kd=np.array([[10.,0.,8.],[0.,10.,8.],[0.,0.,1.]]),scale=1000.)
        views.get=lambda fid:(np.eye(4),depth,np.ones((16,16)))
        measured=np.tile([0.,0.,1.],(128,1))
        self.assertTrue(free_space(measured,views,['a','b','c'])[0])
        floating=measured.copy();floating[:,2]=.8
        self.assertFalse(free_space(floating,views,['a','b','c'])[0])

    def test_visual_part_needs_identity_parent_votes_and_measured_contact(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,12),np.linspace(-.5,.5,12))
        panel=np.column_stack([np.full(y.size,.2),y.ravel(),z.ravel()])
        body=np.concatenate([panel,panel*np.array([-1,1,1])])
        part=dict(name='cabinet: door',status='confirmed',parent_id='b',
            parent_evidence={'b':8},parent_evidence_frames=10,observed_frames=list(range(10)))
        self.assertIsNotNone(part_acceptance(part,panel,body,'door','cabinet',list(range(10))))
        self.assertIsNone(part_acceptance(part,panel,body,'plate','cabinet',list(range(10))))
        self.assertIsNone(part_acceptance({**part,'status':'provisional'},panel,body,'door','cabinet',list(range(10))))
        self.assertIsNone(part_acceptance({**part,'parent_evidence_frames':0},panel,body,'door','cabinet',list(range(10))))
        self.assertIsNone(part_acceptance(part,panel+np.array([.3,0,0]),body,'door','cabinet',list(range(10))))

    def test_handle_extremity_does_not_disprove_shared_measured_body_face(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,20),np.linspace(-.5,.5,20))
        panel=np.column_stack([np.full(y.size,.2),y.ravel(),z.ravel()])
        body=np.concatenate([panel,panel*np.array([-1,1,1]),np.array([[.27,0.,0.]])])
        self.assertIsNone(boundary_relation(panel,body,np.eye(3),np.zeros(3),.95,[1,2,3]))
        actual=boundary_relation(panel,body,np.eye(3),np.zeros(3),.95,[1,2,3],measured_face=True)
        self.assertIsNotNone(actual)
        self.assertIsNotNone(actual['measured_body_face'])
        unrelated=panel.copy();unrelated[:,0]=0.
        self.assertIsNone(boundary_relation(unrelated,body,np.eye(3),np.zeros(3),.95,[1,2,3],measured_face=True))


if __name__=='__main__':unittest.main()
