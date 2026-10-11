"""CPU implementation of the sole PointOps inference operation: exact KNN.

ESAM's official CUDA wrapper returns int32 indices and Euclidean distances.
The model uses k=1 only, for transferring masks to reconstructed vertices.
No model feature/weight/scoring operation is replaced.
"""
import numpy as np,torch
from scipy.spatial import cKDTree

def knn_query(nsample,xyz,offset,new_xyz=None,new_offset=None):
    if new_xyz is None:new_xyz=xyz
    if new_offset is None:new_offset=offset
    source=xyz.detach().cpu().numpy();target=new_xyz.detach().cpu().numpy()
    source_offsets=np.asarray(offset.detach().cpu()).reshape(-1).astype(int)
    target_offsets=np.asarray(new_offset.detach().cpu()).reshape(-1).astype(int)
    assert len(source_offsets)==len(target_offsets)
    previous_source=previous_target=0;indices=[];distances=[]
    for source_end,target_end in zip(source_offsets,target_offsets):
        tree=cKDTree(source[previous_source:source_end])
        dist,idx=tree.query(target[previous_target:target_end],k=nsample,workers=2)
        indices.append(np.asarray(idx).reshape(-1,nsample)+previous_source)
        distances.append(np.asarray(dist).reshape(-1,nsample))
        previous_source,previous_target=source_end,target_end
    return (torch.as_tensor(np.concatenate(indices),dtype=torch.int32,device=xyz.device),
            torch.as_tensor(np.concatenate(distances),dtype=torch.float32,device=xyz.device))

def verify():
    torch.manual_seed(0)
    source=torch.randn(700,3);target=torch.randn(90,3)
    index,distance=knn_query(1,source,torch.tensor([300,700]),target,torch.tensor([40,90]))
    for sb,se,tb,te in [(0,300,0,40),(300,700,40,90)]:
        expected_distance,expected_index=torch.cdist(target[tb:te],source[sb:se]).min(1)
        assert torch.equal(index[tb:te,0].long(),expected_index+sb)
        assert torch.allclose(distance[tb:te,0],expected_distance,atol=2e-5)
    return dict(operation='batched exact Euclidean 1NN',brute_force_index_agreement=1.,
        tolerance=2e-5,CUDA_inference_operations_replaced=['final reconstructed-mask transfer'],
        weights_or_architecture_changed=False)
