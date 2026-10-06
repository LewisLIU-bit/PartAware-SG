"""Verify component identity and ownership prerequisites for measured geometry."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pipeline_components.part_geometry import eligible
class MeasuredPartChecks(unittest.TestCase):
    def test_wrong_parent_semantics_cannot_expand_object(self):
        part=dict(name='sink: basin',status='confirmed',parent_id='1',observed_frames=['a','b','c'],parent_evidence={'1':3},parent_evidence_frames=3)
        self.assertTrue(eligible(part,'1','sink'))
        self.assertFalse(eligible(part,'1','stool'))
        self.assertFalse(eligible(part,'2','sink'))
    def test_inadequate_owner_votes_cannot_expand_object(self):
        part=dict(name='sink: basin',status='confirmed',parent_id='1',observed_frames=['a','b','c'],parent_evidence={'1':3},parent_evidence_frames=7)
        self.assertFalse(eligible(part,'1','sink'))

    def test_publication_preserves_unrelated_completed_points(self):
        import tempfile
        import numpy as np
        import open3d as o3d
        from pipeline_components.part_geometry import replace_regions
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'completed.ply'
            cloud = o3d.geometry.PointCloud()
            other = np.array([[0.,1.,2.],[0.,1.,3.],[0.,1.,4.]])
            cloud.points = o3d.utility.Vector3dVector(np.vstack([[1.,0.,0.],other]))
            cloud.colors = o3d.utility.Vector3dVector(np.array([[1,0,0],[2,0,0],[2,0,0],[2,0,0]])/255)
            o3d.io.write_point_cloud(str(path),cloud)
            replacement = np.array([[1.,0.,1.],[1.,0.,2.]])
            replace_regions(path,{'1':replacement},['1','2'])
            result = o3d.io.read_point_cloud(str(path))
            ids = np.rint(np.asarray(result.colors)[:,0]*255).astype(int)
            np.testing.assert_array_equal(np.asarray(result.points)[ids == 2],other)
            np.testing.assert_array_equal(np.asarray(result.points)[ids == 1],replacement)
    def test_measured_part_requires_three_depth_consistent_mask_views(self):
        import tempfile,json
        import numpy as np
        from pipeline_components.part_geometry import verify
        class Views:
            kc = np.eye(3)
            def get(self,fid):
                return np.eye(4),None,None
            def project(self,points,fid):
                indices = np.arange(len(points)) if fid != 'occluded' else np.array([],int)
                return indices,np.ones(len(indices)),None
        points = np.tile([0.,0.,1.],(16,1))
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for fid in ['a','b','c','occluded']:
                (directory/f'{fid}.json').write_text(json.dumps([dict(track_id='part_1',mask_index=0)]))
                np.savez(directory/f'{fid}.npz',masks=np.ones((1,1,1),bool))
            self.assertEqual(len(verify(points,Views(),['a','b','occluded'],directory,{'part_1'})[0]),0)
            self.assertEqual(len(verify(points,Views(),['a','b','c'],directory,{'part_1'})[0]),16)
            self.assertEqual(len(verify(points,Views(),['a','b','c'],directory,{'other_part'})[0]),0)
