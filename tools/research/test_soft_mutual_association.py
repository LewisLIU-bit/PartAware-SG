"""R2 retains partial measured evidence without confusing it with complete shape."""
from types import SimpleNamespace as NS
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scannet/script'))
from soft_mutual_association import visibility_score,score
grid=np.linspace(-.2,.2,15);xx,yy=np.meshgrid(grid,grid)
points=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]
mask=np.zeros((128,128),bool);mask[42:87,42:87]=True
frame=dict(pose=np.eye(4),kd=np.array([[100,0,64],[0,100,64],[0,0,1.]]),
 kc=np.array([[100,0,64],[0,100,64],[0,0,1.]]),
 depth=np.full((128,128),1000,np.uint16),scale=1000.)
color=np.ones((3,8))/8;semantic=np.array([1.,0.])
track=NS(points=points.copy(),color=color,semantic=semantic)
ob=NS(record={},points=points,mask=mask,color=color,semantic=semantic)
full=visibility_score(ob,track,frame);assert full is not None
fragment=points[0]+np.c_[np.linspace(0,.002,20),np.zeros(20),np.zeros(20)]
small=NS(record={},points=fragment,mask=mask,color=color,semantic=semantic)
partial=visibility_score(small,track,frame)
assert partial is not None and partial['visible_surface_coverage'][1]<.25
assert score(2,full)>score(2,partial)
# Geometrically ambiguous identities retain the language feature as a soft cue,
# including different names in the same open vocabulary, never a category gate.
different=NS(points=points.copy(),color=color,semantic=np.array([0.,1.]))
other=visibility_score(ob,different,frame)
assert other is not None and score(2,full)>score(2,other)
hidden=points.copy();hidden[:,2]=2
back=visibility_score(ob,NS(points=np.r_[points,hidden],color=color,semantic=semantic),frame)
assert back is not None and score(2,back)==score(2,full)
print('SOFT_MUTUAL_TEST_PASSED: partial retained with lower score, semantic ranking, hidden back unknown')
