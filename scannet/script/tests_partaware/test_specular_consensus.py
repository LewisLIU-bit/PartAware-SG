"""Check that flat geometry alone and shiny objects do not justify rejection."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from pipeline_components.specular_consensus import measure,rejection
class SurfaceChecks(unittest.TestCase):
    def test_flat_sink_requires_independent_background(self):
        class Views:
            jobs={str(i):{} for i in range(8)}
            records={str(i):[] for i in range(8)}
            def project(self,p,fid):
                return np.arange(len(p)),np.ones(len(p)),np.zeros(len(p),int)
        p=np.column_stack([np.linspace(0,.5,100),np.sin(np.arange(100))*.2,np.ones(100)])
        e=measure('sink',p,Views(),upright=True)
        self.assertTrue(rejection(e))
        e=measure('sink',p,Views(),observed_frames=list(Views.jobs),upright=True)
        self.assertFalse(rejection(e))
    def test_known_axes_and_concave_semantics_are_required(self):
        p=np.zeros((100,3))
        self.assertFalse(measure('plate',p,object(),upright=True)['applicable'])
        self.assertFalse(measure('sink',p,object(),upright=False)['applicable'])
    def test_curved_surface_is_protected(self):
        rng=np.random.default_rng(2);p=rng.normal(size=(100,3))*.1
        self.assertFalse(measure('sink',p,object(),upright=True)['applicable'])
