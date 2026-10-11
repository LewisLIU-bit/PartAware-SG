from types import SimpleNamespace
import torch
from torch import nn
from freeze_adapter import EviSceneFreezeHook
model=nn.Module();model.backbone=nn.Sequential(nn.Linear(4,4),nn.BatchNorm1d(4));model.decoder=nn.Linear(4,1)
runner=SimpleNamespace(model=model)
hook=EviSceneFreezeHook();hook.before_train(runner)
assert not any(p.requires_grad for p in model.backbone.parameters())
assert all(p.requires_grad for p in model.decoder.parameters())
before=model.backbone[1].running_mean.clone()
model.train();hook.before_train_iter(runner,0)
model.decoder(model.backbone(torch.randn(8,4))).sum().backward()
assert torch.equal(before,model.backbone[1].running_mean)
assert model.decoder.weight.grad is not None
print('FREEZE_PARAMETERS_AND_NORM_BUFFERS_PASSED')
