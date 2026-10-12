"""Memory-bounded, behavior-equivalent signed CORE for this screen.

Exclusive native masks are encoded as one owner vector. No changed scores,
thresholds, NMS or core-distance policy. A regression test compares every mask
and score against the original signed implementation on overlapping fixtures.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

def intersections(native_owner,count,peer):
    owner=np.asarray(native_owner,int);b=np.asarray(peer,bool)
    mass=np.bincount(owner[owner>=0],minlength=count)
    cross=np.empty((count,len(b)),np.int64)
    for j,mask in enumerate(b):cross[:,j]=np.bincount(owner[mask]+1,minlength=count+1)[1:]
    return cross,mass

def propose(native_owner,native_scores,peer,peer_scores,points):
    owner=np.asarray(native_owner,int);sa=np.asarray(native_scores,float)
    b=np.asarray(peer,bool);sb=np.asarray(peer_scores,float);xyz=np.asarray(points,float)
    cross,ma=intersections(owner,len(sa),b);mb=b.sum(1)
    union=ma[:,None]+mb[None,:]-cross
    ab=np.divide(cross,union,out=np.zeros_like(union,dtype=float),where=union>0)
    qa=.5*(sa+(ab.max(1) if len(b) else 0));qb=.5*(sb+(ab.max(0) if len(sa) else 0))
    scores=np.r_[qa,qb];mass=np.r_[ma,mb];total=len(scores);na=len(sa)
    overlap=np.zeros((total,total));overlap[:na,:na]=np.diag((ma>0).astype(float))
    overlap[:na,na:]=ab;overlap[na:,:na]=ab.T
    sparse=csr_matrix(b,dtype=np.int64);bb=(sparse@sparse.T).toarray();del sparse
    u=mb[:,None]+mb[None,:]-bb
    overlap[na:,na:]=np.divide(bb,u,out=np.zeros_like(u,dtype=float),where=u>0)
    keep=[];suppressed=[]
    for index in np.lexsort((-np.arange(total),-scores)):
        if mass[index]<100:continue
        if keep and np.any(overlap[index,keep]>.5):suppressed.append(int(index));continue
        keep.append(int(index))
    def mask(index):return owner==index if index<na else b[index-na]
    multiplicity=np.zeros(len(owner),np.int32)
    for index in keep:multiplicity+=mask(index)
    output=np.full(len(owner),-1,np.int32);best=np.full(len(owner),-np.inf);cores=[]
    for rank,index in enumerate(keep):
        current=mask(index);vertices=np.flatnonzero(current)
        core=xyz[current&(multiplicity==1)];cores.append(len(core))
        if len(core)<16:core=xyz[current]
        distance=cKDTree(core).query(xyz[vertices],workers=2)[0]
        value=scores[index]/(1+distance/.05);wins=value>best[vertices];chosen=vertices[wins]
        output[chosen]=rank;best[chosen]=value[wins]
    return output,scores[keep],dict(kept=keep,suppressed=suppressed,uncontested_core_sizes=cores,
        scores_calibrated=False,policy='unchanged_signed_CORE',exclusive_point_ownership=True)

def smooth(labels,segments):
    _,group=np.unique(segments,return_inverse=True);count=np.bincount(group)
    output=np.full(labels.shape,-1,dtype=labels.dtype)
    for label in np.unique(labels):
        if label<0:continue
        positive=np.bincount(group[labels==label],minlength=len(count))
        output[(positive>count*.5)[group]]=label
    return output
