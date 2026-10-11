"""Restore TensorField point-feature/coordinate alignment on ME CPU.

ME 0.5.4's CPU coordinate hash map reports field rows in hash order.
Official ESAM concatenates TensorField.decomposed_features with original
input XYZ and predicts point masks in input order. Sorting field row indices
restores that same point order for every batch, independent of scene or GT.
SparseTensor voxel order and all model calculations/weights remain intact.
"""
import torch,MinkowskiEngine as ME

_original_field_rows=ME.TensorField._batchwise_row_indices

def install():
    if getattr(ME.TensorField,'_esam_cpu_order_fixed',False):return
    def ordered_rows(field):
        return [torch.sort(rows).values for rows in _original_field_rows.fget(field)]
    ME.TensorField._batchwise_row_indices=property(ordered_rows)
    ME.TensorField._esam_cpu_order_fixed=True

def verify():
    torch.manual_seed(0)
    xyz=torch.randn(20000,3);features=torch.arange(20000,dtype=torch.float32).view(-1,1)
    c,f=ME.utils.batch_sparse_collate([(xyz/.02,features)],device='cpu')
    field=ME.TensorField(coordinates=c,features=f)
    output=field.sparse().slice(field)
    assert torch.equal(output._batchwise_row_indices[0],torch.arange(20000))
    assert torch.equal(output.features,output.decomposed_features[0])
    assert torch.equal(output.C[:,1:],output.decomposed_coordinates[0])
    return dict(point_count=20000,field_feature_coordinate_order_verified=True,
        scope='All CPU TensorField batches; no scene/label/GT condition',
        model_architecture_or_weights_changed=False)
